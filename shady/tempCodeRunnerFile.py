import re

# Your specific file path
file_path = r"C:\Users\sanji\Downloads\WhatsApp Chat with Luminous.txt"
sender_name = "Sundance"
output_file = r"C:\Users\sanji\Downloads\Sundance_messages.txt"

def extract_messages(file_path, sender_name):
    """
    Extract messages from a specific sender in a WhatsApp chat export
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # Try with another encoding if utf-8 fails
        with open(file_path, 'r', encoding='utf-16') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Pattern to match WhatsApp messages with timestamps and sender names
    pattern = r'\d+/\d+/\d+,\s\d+:\d+\s[AP]M\s-\s' + re.escape(sender_name) + r':\s(.*?)(?=\n\d+/\d+/\d+|\Z)'
    
    # If the pattern doesn't match, try alternative format (24-hour time)
    if not re.findall(pattern, content, re.DOTALL):
        pattern = r'\d+/\d+/\d+,\s\d+:\d+\s-\s' + re.escape(sender_name) + r':\s(.*?)(?=\n\d+/\d+/\d+|\Z)'
    
    messages = []
    
    for match in re.finditer(pattern, content, re.DOTALL):
        message = match.group(1).strip()
        if message.lower() != 'null':  # Skip 'null' messages
            messages.append(message)
    
    return messages

# Main execution
print(f"Extracting messages from: {file_path}")
print(f"Looking for messages from: {sender_name}")

messages = extract_messages(file_path, sender_name)

if not messages:
    print(f"No messages found from {sender_name} or file couldn't be read correctly.")
else:
    # Save messages to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for message in messages:
            file.write(message + '\n')
    
    print(f"\nExtracted {len(messages)} messages from {sender_name}.")
    print(f"Messages saved to {output_file}")
    print("\nFirst few messages:")
    for message in messages[:5]:
        print(f"- {message}")

print("\nPress Enter to exit...")
input()

#this is used for whatsapp saved message processing
