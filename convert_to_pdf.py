
import markdown_pdf

def convert_readme_to_pdf():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            md_content = f.read()

        pdf = markdown_pdf.MarkdownPdf()
        pdf.add_section(md_content)
        pdf.save("README.pdf")
        print("Successfully converted README.md to README.pdf")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_readme_to_pdf()
