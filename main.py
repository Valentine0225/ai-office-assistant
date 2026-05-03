import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from prompts import SUMMARY_PROMPT, REPORT_PROMPT, MEETING_PROMPT


def get_client() -> OpenAI:
    """Create and return an OpenAI-compatible client."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def get_model_name() -> str:
    """Return model name from env or use a default."""
    return os.getenv("OPENAI_MODEL", "qwen2.5:7b-instruct")


def call_llm(system_prompt: str, user_input: str) -> str:
    """Send prompt to model and return the response text."""
    client = get_client()
    model = get_model_name()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def summarize_text(text: str) -> str:
    """Summarize input text."""
    return call_llm(SUMMARY_PROMPT, text)


def generate_report(text: str) -> str:
    """Generate a daily report from user notes."""
    return call_llm(REPORT_PROMPT, text)


def generate_meeting_minutes(text: str) -> str:
    """Generate structured meeting minutes."""
    return call_llm(MEETING_PROMPT, text)


def read_text_file(file_path: str) -> str:
    """Read text content from a .txt file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.endswith(".txt"):
        raise ValueError("Only .txt files are supported right now.")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_input_content(raw_input: str) -> str:
    """Return raw text or file content."""
    if raw_input.endswith(".txt"):
        return read_text_file(raw_input)
    return raw_input


def print_usage() -> None:
    """Print command usage."""
    usage = """
Usage:
  python main.py summarize "这里放要总结的文本"
  python main.py report "这里放今天做的事情"
  python main.py meeting "这里放会议记录"
  python main.py summarize notes.txt
  python main.py meeting meeting_notes.txt

Examples:
  python main.py summarize "今天我们讨论了无人车项目的测试流程优化..."
  python main.py report "今天学习了Python，完成了AI总结工具的原型开发，修复了一个报错"
  python main.py meeting "今天开会讨论了项目分工，张三负责前端，李四负责后端，下周一前完成第一版"
"""
    print(usage)


def main() -> None:
    """Entry point for CLI tool."""
    if len(sys.argv) < 3:
        print_usage()
        return

    command = sys.argv[1].strip().lower()
    raw_input = " ".join(sys.argv[2:]).strip()

    if not raw_input:
        print("Error: input text is empty.")
        return

    try:
        content = get_input_content(raw_input)

        if command == "summarize":
            result = summarize_text(content)
        elif command == "report":
            result = generate_report(content)
        elif command == "meeting":
            result = generate_meeting_minutes(content)
        else:
            print(f"Unknown command: {command}")
            print_usage()
            return

        print("\n===== RESULT =====\n")
        print(result)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()