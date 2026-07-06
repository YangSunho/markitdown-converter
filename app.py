"""
MarkItDown 변환기
PDF(또는 기타 지원 파일)를 마크다운으로 변환합니다.

사용법:
  1) GUI 모드: 인자 없이 실행하면 파일 탐색기 창이 뜹니다.
     python app.py

  2) 커맨드라인 모드: 입력 파일과 출력 폴더를 직접 지정합니다.
     python app.py --input "파일경로.pdf" --output "저장할폴더"

  3) LLM(OpenAI)로 이미지/스캔본 해석이 필요하면 --use-llm 옵션을 추가합니다.
     (openai 패키지 설치 및 OPENAI_API_KEY 환경변수 필요)
     python app.py --input "note.jpg" --output "저장할폴더" --use-llm
"""

import argparse
import os
import sys

from markitdown import MarkItDown


def pick_input_file() -> str:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="변환할 파일을 선택하세요",
        filetypes=[
            (
                "지원 파일",
                "*.pdf *.docx *.pptx *.xls *.xlsx *.csv *.png *.jpg *.jpeg "
                "*.html *.htm *.txt *.epub *.msg *.ipynb *.zip "
                "*.mp3 *.wav *.m4a",
            ),
            ("모든 파일", "*.*"),
        ],
    )
    root.destroy()
    return path


def pick_output_dir() -> str:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="결과물을 저장할 폴더를 선택하세요")
    root.destroy()
    return path


def convert(input_path: str, output_dir: str, use_llm: bool = False) -> str:
    if not input_path or not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    if not output_dir:
        raise ValueError("출력 폴더가 지정되지 않았습니다.")

    os.makedirs(output_dir, exist_ok=True)

    if use_llm:
        from openai import OpenAI

        md = MarkItDown(llm_client=OpenAI(), llm_model="gpt-4o")
    else:
        md = MarkItDown()

    result = md.convert(input_path)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.markdown)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="MarkItDown 변환기")
    parser.add_argument("--input", "-i", help="변환할 파일 경로")
    parser.add_argument("--output", "-o", help="결과물을 저장할 폴더 경로")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="OpenAI(gpt-4o)를 사용해 이미지/스캔본을 해석합니다 (openai 패키지 필요)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    if not input_path:
        input_path = pick_input_file()
    if not output_dir:
        output_dir = pick_output_dir()

    if not input_path or not output_dir:
        print("입력 파일 또는 출력 폴더가 선택되지 않았습니다. 종료합니다.")
        sys.exit(1)

    print(f"입력 파일: {input_path}")
    print(f"출력 폴더: {output_dir}")
    print("변환 중...")

    output_path = convert(input_path, output_dir, use_llm=args.use_llm)

    print(f"완료! 결과물이 저장되었습니다: {output_path}")


if __name__ == "__main__":
    main()
