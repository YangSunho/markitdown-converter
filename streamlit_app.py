import base64
import io
import os
import re
import time
import zipfile

import pypdfium2 as pdfium
import requests
import streamlit as st
from markitdown import MarkItDown

APP_VERSION = "1.0.0"
APP_DEV_DATE = "2026-07-06"
APP_AUTHOR = "양선호 (Yang, Sunho)"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Word/HTML 변환 경로는 문서 안 이미지를 캡션 없이 base64 데이터 URI로 그대로
# 남기므로(마크다운 이미지 문법 `![](data:image/...;base64,...)`), LLM이 켜져
# 있으면 이 패턴을 찾아 이미지 내용을 직접 해석한 텍스트로 바꿔준다.
IMG_DATA_URI_RE = re.compile(
    r"!\[[^\]]*\]\(data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)\)"
)


def describe_embedded_images(markdown_text: str, client, model: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        mimetype, b64data = match.group(1), match.group(2)
        data_uri = f"data:{mimetype};base64,{b64data}"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "이 이미지에 담긴 내용을 한국어로 상세히 설명해줘. "
                                    "표나 텍스트가 보이면 최대한 그대로 옮겨써줘."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            )
            caption = response.choices[0].message.content
        except Exception as e:
            caption = f"(이미지 해석 실패: {e})"
        return f"\n\n> 🖼️ **[이미지 해석]** {caption}\n\n"

    return IMG_DATA_URI_RE.sub(_replace, markdown_text)


# MarkItDown의 PdfConverter는 pdfplumber/pdfminer로 텍스트·표만 추출하고,
# 문서 안의 이미지/차트/도표는 아예 마크다운에 남기지 않는다(추출 자체를
# 안 함). 그래서 describe_embedded_images()로는 PDF 안 이미지를 찾을 수
# 없다 — PDF 원본 파일 전체를 Gemini의 네이티브 문서이해 API로 직접 보내
# "이 안에 이미지/도표가 있으면 설명해줘"라고 따로 요청해야 한다. (Gemini의
# OpenAI 호환 엔드포인트는 이미지만 지원하고 PDF 문서 입력은 지원하지 않아
# 네이티브 REST 엔드포인트를 그대로 사용한다.)
GEMINI_NATIVE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def describe_pdf_visuals(pdf_bytes: bytes, api_key: str, model: str) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "application/pdf", "data": b64}},
                    {
                        "text": (
                            "이 PDF 문서(또는 페이지) 안에서 텍스트가 아니라 이미지, 차트, "
                            "도표, 플로우차트 등 시각 자료로 들어가 있는 내용을 찾아서, 각각 "
                            "무엇을 나타내는지 한국어로 상세히 설명해줘. 표나 텍스트가 이미지 "
                            "안에 있으면 최대한 그대로 옮겨써줘. 텍스트가 아닌 별도의 이미지/"
                            "도표가 전혀 없다면, 다른 말 없이 정확히 NONE 이라고만 답해줘."
                        )
                    },
                ]
            }
        ]
    }
    resp = requests.post(
        GEMINI_NATIVE_URL.format(model=model),
        params={"key": api_key},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def split_pdf_page(pdf_bytes: bytes, page_index: int) -> bytes:
    src_doc = pdfium.PdfDocument(pdf_bytes)
    single_doc = pdfium.PdfDocument.new()
    single_doc.import_pages(src_doc, pages=[page_index])
    buf = io.BytesIO()
    single_doc.save(buf)
    return buf.getvalue()


def detect_visual_pages(pdf_bytes: bytes) -> list[bool]:
    """
    각 페이지에 그림/도형/이미지가 있는지 pdfplumber로 로컬에서(무료·빠름) 판별한다.
    삽입 이미지, 사각형, 선, 곡선 중 하나라도 있으면 시각 요소가 있는 것으로 본다
    (보수적 판별 — 100% 순수 텍스트 페이지만 걸러낸다). 직선·사각형으로 그린 박스나
    도표도 놓치지 않으므로, 곡선만 보는 방식보다 안전하다. 반환값은 페이지별 True/False.
    """
    import pdfplumber

    flags: list[bool] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            has_visual = bool(
                page.images or page.rects or page.lines or page.curves
            )
            flags.append(has_visual)
    return flags


def convert_pdf_with_inline_visuals(
    pdf_bytes: bytes, md: MarkItDown, api_key: str, model: str
) -> str:
    """
    PDF를 페이지 단위로 쪼개서 각 페이지를 개별 변환하고, 그 페이지의 이미지/도표
    해석(있는 경우)을 페이지 텍스트 바로 뒤에 붙인다. 문서 전체를 한 번에 보내는
    방식과 달리, 이미지 해석이 문서 맨 끝으로 밀리지 않고 원래 위치 근처에 남는다.

    속도 최적화: 그림/도형이 있는 페이지만 Gemini에 보내고, 순수 텍스트 페이지는
    호출과 대기를 모두 건너뛴다.
    """
    visual_flags = detect_visual_pages(pdf_bytes)
    num_pages = len(visual_flags)
    page_sections = []
    called_gemini_once = False
    for page_idx in range(num_pages):
        single_page_bytes = split_pdf_page(pdf_bytes, page_idx)

        page_result = md.convert_stream(
            io.BytesIO(single_page_bytes), file_extension=".pdf"
        )
        section = page_result.markdown.strip()

        if visual_flags[page_idx]:
            # Gemini 무료 등급 분당 요청 한도(RPM)를 넘기지 않도록 실제 호출
            # 사이에만 잠깐 대기 (텍스트 전용 페이지는 대기 없이 넘어간다).
            if called_gemini_once:
                time.sleep(4)
            called_gemini_once = True

            try:
                visual = describe_pdf_visuals(single_page_bytes, api_key, model).strip()
            except Exception as e:
                visual = f"(페이지 {page_idx + 1} 이미지 해석 실패: {e})"

            if visual and visual != "NONE":
                section += (
                    f"\n\n> 🖼️ **[페이지 {page_idx + 1} 이미지/도표 해석]** {visual}"
                )

        page_sections.append(section)

    return "\n\n".join(s for s in page_sections if s)


st.set_page_config(page_title="MarkItDown 변환기", page_icon="📄", layout="centered")

# DESIGN.md 토큰 적용: 근흑(ink) 프라이머리 버튼, 헤어라인 인풋, 12px/6px 라운딩,
# 링크 블루, 뮤트 캡션 컬러 — 마케팅 신호 카드류는 이 유틸리티 앱 성격과 맞지 않아 제외.
st.markdown(
    """
    <style>
    .stButton > button[kind="primary"], .stDownloadButton > button {
        background-color: #181d26;
        color: #ffffff;
        border-radius: 12px;
        border: none;
    }
    .stButton > button[kind="primary"]:active, .stDownloadButton > button:active {
        background-color: #0d1218;
    }
    .stTextInput input {
        border-radius: 6px;
    }
    div[data-baseweb="input"] {
        border-radius: 6px;
    }
    a { color: #1b61c9; }
    [data-testid="stCaptionContainer"] { color: #41454d; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.dialog("📖 사용법")
def show_help_dialog():
    st.markdown(
        """
        1. **파일 선택**: 변환할 파일(PDF, Word, PowerPoint, Excel, 이미지 등)을
           업로드하세요. 여러 개를 한 번에 선택할 수 있습니다.
        2. **(선택) 이미지/스캔본 해석**: 손글씨 메모, 스캔본, 문서에 박힌 이미지의
           내용까지 읽고 싶다면 "이미지/스캔본 해석에 Google Gemini 사용"을 켜고
           Gemini API Key를 입력하세요. 켜지 않으면 이미지 속 내용은 인식되지 않습니다.
        3. **변환하기** 버튼을 누르면 자동으로 마크다운(.md)으로 변환됩니다.
        4. 변환이 끝나면 파일별 **다운로드** 버튼으로 결과를 받을 수 있고,
           여러 파일을 올렸다면 **전체 ZIP 다운로드**로 한 번에 받을 수도 있습니다.
        5. 다운로드할 때마다 폴더를 직접 고르고 싶다면, 브라우저 설정에서
           "다운로드 전에 저장 위치 확인"을 켜두면 됩니다.
        6. 위 방법으로도 잘 안 되면 **fineyang@gmail.com** 으로 메일 문의해주세요.
        """
    )


st.title("📄 MarkItDown 변환기")
st.caption("PDF, Word, PowerPoint, Excel, 이미지 등 다양한 파일을 마크다운으로 변환합니다.")
st.caption(
    f"제작: **{APP_AUTHOR}** · v{APP_VERSION} · {APP_DEV_DATE} · "
    "변환 엔진: [Microsoft MarkItDown](https://github.com/microsoft/markitdown) (MIT License)"
)
with st.sidebar:
    st.header("설정")
    use_llm = st.checkbox("이미지/스캔본 해석에 Google Gemini 사용", value=False)
    api_key = None
    llm_model = "gemini-2.5-flash"
    if use_llm:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="https://aistudio.google.com/apikey 에서 무료로 발급받을 수 있습니다 "
            "(신용카드 등록 불필요).",
        )
        llm_model = st.text_input(
            "모델명",
            value="gemini-2.5-flash",
            help="Gemini 계정에서 사용 가능한 비전(이미지 인식) 모델명을 입력하세요. "
            "무료 등급 할당량이 0으로 나오면 다른 모델명(예: gemini-2.5-flash-lite)으로 "
            "바꿔서 시도해보세요. Google이 모델을 새로 내놓으면 여기에 최신 모델명을 넣으면 됩니다.",
        )
        st.caption("손글씨나 스캔 이미지처럼 텍스트 추출이 어려운 파일을 위 모델로 해석합니다.")
        st.caption(
            "📄 PDF는 그림/도형/표가 있는 페이지만 골라 해석하고, 그 결과를 해당 "
            "페이지 바로 뒤에 넣습니다. 순수 텍스트 페이지는 건너뛰어 빠릅니다. "
            "그림이 많은 문서는 시간이 더 걸릴 수 있습니다."
        )

    st.divider()
    with st.expander("ℹ️ 앱 정보"):
        st.markdown(
            f"""
            - **제작자**: {APP_AUTHOR}
            - **버전**: v{APP_VERSION}
            - **개발일자**: {APP_DEV_DATE}
            - **변환 엔진**: [Microsoft MarkItDown](https://github.com/microsoft/markitdown)

            이 앱은 Microsoft가 공개한 오픈소스 라이브러리
            **MarkItDown**(MIT License)을 이용해 파일을
            마크다운으로 변환하는 웹 UI를 씌운 것입니다.
            """
        )

    if st.button("📖 사용법 보기"):
        show_help_dialog()

uploaded_files = st.file_uploader(
    "변환할 파일을 선택하세요 (여러 개 선택 가능)",
    type=[
        "pdf", "docx", "pptx", "xls", "xlsx", "csv",
        "png", "jpg", "jpeg", "html", "htm", "txt",
        "epub", "msg", "ipynb", "zip", "mp3", "wav", "m4a",
    ],
    accept_multiple_files=True,
)

convert_clicked = st.button("변환하기", type="primary", disabled=not uploaded_files)

if convert_clicked:
    if use_llm and not api_key:
        st.error("Google Gemini API Key를 입력해주세요.")
    elif use_llm and not llm_model:
        st.error("모델명을 입력해주세요.")
    else:
        llm_client_obj = None
        if use_llm:
            from openai import OpenAI
            llm_client_obj = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
            md = MarkItDown(llm_client=llm_client_obj, llm_model=llm_model)
        else:
            md = MarkItDown()

        results = {}
        progress = st.progress(0.0)
        for i, uf in enumerate(uploaded_files):
            try:
                ext = os.path.splitext(uf.name)[1]

                if ext.lower() == ".pdf" and llm_client_obj is not None:
                    markdown_text = convert_pdf_with_inline_visuals(
                        uf.getvalue(), md, api_key, llm_model
                    )
                else:
                    stream = io.BytesIO(uf.getvalue())
                    # keep_data_uris: LLM으로 이미지를 해석하려면 문서에 박힌 이미지의
                    # 실제 base64 데이터가 markdown에 남아있어야 한다 (기본값은 잘림).
                    result = md.convert_stream(
                        stream, file_extension=ext, keep_data_uris=use_llm
                    )
                    markdown_text = result.markdown
                    if llm_client_obj is not None:
                        markdown_text = describe_embedded_images(
                            markdown_text, llm_client_obj, llm_model
                        )
                results[uf.name] = markdown_text
            except Exception as e:
                st.error(f"'{uf.name}' 변환 실패: {e}")
            progress.progress((i + 1) / len(uploaded_files))

        st.session_state["results"] = results

if st.session_state.get("results"):
    results = st.session_state["results"]
    st.success(f"{len(results)}개 파일 변환 완료")

    if len(results) > 1:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, content in results.items():
                base_name = os.path.splitext(name)[0]
                zf.writestr(f"{base_name}.md", content)
        st.download_button(
            "전체 결과 ZIP으로 다운로드",
            zip_buffer.getvalue(),
            file_name="converted_markdown.zip",
            mime="application/zip",
        )

    for i, (name, content) in enumerate(results.items()):
        base_name = os.path.splitext(name)[0]
        col_name, col_download = st.columns([4, 1])
        with col_name:
            st.write(f"📄 {name}")
        with col_download:
            st.download_button(
                "다운로드",
                content,
                file_name=f"{base_name}.md",
                mime="text/markdown",
                key=f"dl_{i}_{name}",
            )
        with st.expander("미리보기"):
            st.markdown(content)

st.divider()
st.caption(
    f"© {APP_DEV_DATE[:4]} {APP_AUTHOR} · MarkItDown 변환기 v{APP_VERSION} ({APP_DEV_DATE}) · "
    "Powered by [Microsoft MarkItDown](https://github.com/microsoft/markitdown) (MIT License)"
)
st.caption(
    "⚠️ 본인이 저작권을 보유했거나 이용 권한이 있는 파일만 업로드해주세요. "
    "업로드한 파일의 저작권 관련 책임은 업로드한 사용자 본인에게 있습니다."
)
