import os
import glob

files = glob.glob('src/*.py')
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace escaped quotes back to normal docstring quotes
    new_content = content.replace('\\"\\"\\"', '"""')
    
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Fixed {f}")
print("Done!")
