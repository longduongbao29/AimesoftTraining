from py_pdf_parser.loaders import load_file

# Step 1 - Load the document
document = load_file("data/simple_memo.pdf")

# We could visualise it here to check it looks correct:
# from py_pdf_parser.visualise import visualise
# visualise(document)

# Step 2 - Extract reference elements:
to_element = document.elements.filter_by_text_equal("TO:").extract_single_element()
from_element = document.elements.filter_by_text_equal("FROM:").extract_single_element()
date_element = document.elements.filter_by_text_equal("DATE:").extract_single_element()
subject_element = document.elements.filter_by_text_equal(
    "SUBJECT:"
).extract_single_element()

# Step 3 - Extract the data
to_text = document.elements.to_the_right_of(to_element).extract_single_element().text()
from_text = (
    document.elements.to_the_right_of(from_element).extract_single_element().text()
)
date_text = (
    document.elements.to_the_right_of(date_element).extract_single_element().text()
)
subject_text_element = document.elements.to_the_right_of(
    subject_element
).extract_single_element()
subject_text = subject_text_element.text()

content_elements = document.elements.after(subject_element)
content_text = "\n".join(element.text() for element in content_elements)

output = {
    "to": to_text,
    "from": from_text,
    "date": date_text,
    "subject": subject_text,
    "content": content_text,
}
print(output)