def clean_text(input_text: str):
    """Remove unwanted lines from the extracted text."""
    lines = input_text.split('\n')
    filtered_lines = [line for line in lines if not (line.startswith("Figure ") or 
                                                     line.startswith(" Page") or 
                                                     line.startswith("Page") or  # Added condition to remove lines starting with "Page"
                                                     line.startswith("Confidential and") or 
                                                     line.startswith("Coil System Specifications"))]
    return '\n'.join(filtered_lines)

if __name__ == "__main__":
    # Read a txt file and clean the text
    with open("documents/TELIA for Coil - Remapping User Guide.txt", "r", encoding="utf-8") as file:
        input_text = file.read()
        cleaned_text = clean_text(input_text)
        # Store the cleaned text back to a file
        with open("documents/TELIA for Coil - Remapping User Guide_cleaned.txt", "w", encoding="utf-8") as cleaned_file:
            cleaned_file.write(cleaned_text)


