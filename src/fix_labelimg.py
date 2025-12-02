import os
import sys
import site

def auto_fix_labelimg():
    print("Searching for labelImg broken file (canvas.py)...")
    
    # Get all possible library paths for the current environment (venv)
    possible_paths = sys.path
    target_file = None

    for p in possible_paths:
        # Check standard install location
        candidate = os.path.join(p, "libs", "canvas.py")
        if os.path.exists(candidate):
            target_file = candidate
            break
        # Check if it's inside a labelImg folder (common in some installs)
        candidate_nested = os.path.join(p, "labelImg", "libs", "canvas.py")
        if os.path.exists(candidate_nested):
            target_file = candidate_nested
            break

    if not target_file:
        print("❌ ERROR: Could not find 'libs/canvas.py'.")
        print("Debug info - Checked these paths:")
        for p in possible_paths:
            print(f" - {p}")
        print("\nMake sure you ran 'pip install labelImg' inside this venv!")
        return

    print(f"✅ Found file at: {target_file}")
    print("Applying fixes for Python 3.12...")

    with open(target_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    fixed_count = 0

    for line in lines:
        # Fix 1: Horizontal Lines
        if "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())" in line:
            new_lines.append("            p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))\n")
            fixed_count += 1
            
        # Fix 2: Vertical Lines
        elif "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())" in line:
            new_lines.append("            p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))\n")
            fixed_count += 1
            
        # Fix 3: Rectangles (The Box Drawing Crash)
        elif "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)" in line:
            new_lines.append("            p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))\n")
            fixed_count += 1
             
        else:
            new_lines.append(line)

    with open(target_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ SUCCESS: Patched {fixed_count} bugs. You can now run labelImg.")

if __name__ == "__main__":
    auto_fix_labelimg()