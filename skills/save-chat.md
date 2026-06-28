# Purpose

This skill set enables an agent to save conversation part specified by the user.

# Instructions

1. From the user's input, infer which parts of the chat are relevant, including the user's prompt. 
2. Remember the contents of all messages within the relevant boundaries.
3. Save the selected messages (both AI and user) into history/ folder in a chronological order.

# Rules

- Save each chat history as a unique markdown file.
- If the user prompt is very long (>500 characters), save a summary that does not exceed 500 characters.