import os
from pathlib import Path
from typing import Set, List

def count_lines_in_file(file_path: str) -> tuple[int, int, int]:
    """
    Count lines in a single file.
    Returns (total_lines, code_lines, comment_lines)
    """
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            in_multiline_comment = False
            
            for line in file:
                total_lines += 1
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Handle multi-line comments for Python
                if line.startswith('"""') or line.startswith("'''"):
                    in_multiline_comment = not in_multiline_comment
                    comment_lines += 1
                    continue
                
                if in_multiline_comment:
                    comment_lines += 1
                    continue
                
                # Single line comments
                if line.startswith('#'):
                    comment_lines += 1
                else:
                    code_lines += 1
                    
        return total_lines, code_lines, comment_lines
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0, 0, 0

def should_skip_dir(dir_name: str, skip_dirs: Set[str]) -> bool:
    """Check if directory should be skipped."""
    return any(skip_dir in dir_name for skip_dir in skip_dirs)

def should_process_file(file_name: str, extensions: List[str]) -> bool:
    """Check if file should be processed based on its extension."""
    return any(file_name.endswith(ext) for ext in extensions)

def count_lines_in_project(
    project_path: str,
    extensions: List[str] = ['.py'],
    skip_dirs: Set[str] = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'build', 'dist'}
) -> tuple[int, int, int]:
    """
    Count lines in all files in the project.
    Returns (total_lines, code_lines, comment_lines)
    """
    total_lines = 0
    total_code_lines = 0
    total_comment_lines = 0
    files_processed = 0
    
    for root, dirs, files in os.walk(project_path):
        # Skip directories we don't want to process
        dirs[:] = [d for d in dirs if not should_skip_dir(d, skip_dirs)]
        
        for file in files:
            if should_process_file(file, extensions):
                file_path = os.path.join(root, file)
                lines, code, comments = count_lines_in_file(file_path)
                
                if lines > 0:
                    rel_path = os.path.relpath(file_path, project_path)
                    print(f"\nFile: {rel_path}")
                    print(f"  Total lines: {lines}")
                    print(f"  Code lines: {code}")
                    print(f"  Comment lines: {comments}")
                    
                    total_lines += lines
                    total_code_lines += code
                    total_comment_lines += comments
                    files_processed += 1
    
    return total_lines, total_code_lines, total_comment_lines, files_processed

if __name__ == "__main__":
    # Get the current working directory
    project_path = os.getcwd()
    
    # Define which file extensions to count
    extensions = ['.py']  # Add more extensions if needed, e.g., ['.py', '.js', '.cpp']
    
    print(f"Analyzing project at: {project_path}")
    print(f"Looking for files with extensions: {extensions}")
    print("-" * 50)
    
    total, code, comments, files = count_lines_in_project(project_path, extensions)
    
    print("\nSummary:")
    print("-" * 50)
    print(f"Files processed: {files}")
    print(f"Total lines: {total}")
    print(f"Code lines: {code}")
    print(f"Comment lines: {comments}")
    
    if total > 0:
        comment_ratio = (comments / total) * 100
        print(f"Comment ratio: {comment_ratio:.1f}%")