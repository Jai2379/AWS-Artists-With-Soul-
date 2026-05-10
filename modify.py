import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Colors
content = content.replace('--bg-dark: #1a0500;', '--bg-dark: #FDF8F5;')
content = content.replace('--bg-darker: #0d0200;', '--bg-darker: #F4EBE6;')
content = content.replace('--text-cream: #FFF8EE;', '--text-cream: #1a0500;')
content = content.replace("--accent-gold: #FFD97D;", "--accent-gold: #D4A017;")

# 2. Replaces rgba(255, 248, 238, ...) with rgba(26, 5, 0, ...)
content = content.replace('rgba(255, 248, 238,', 'rgba(26, 5, 0,')

# 3. Fix nav background gradient from dark to light
content = content.replace('rgba(26, 5, 0, 0.9)', 'rgba(253, 248, 245, 0.9)')
content = content.replace('rgba(26, 5, 0, 0)', 'rgba(253, 248, 245, 0)')

# 4. Remove cursor CSS
content = re.sub(r'\s*cursor:\s*none;\n?', '\n', content)
content = re.sub(r'#cursor-dot\s*\{[^}]+\}\s*#cursor-dot\.hovering\s*\{[^}]+\}', '', content)
content = content.replace('<div id="cursor-dot"></div>\n', '')

# 5. Fix Javascript Cursor Dot
content = re.sub(r"const cursorDot = document\.getElementById\('cursor-dot'\);.*?document\.addEventListener\('mousemove'", "document.addEventListener('mousemove'", content, flags=re.DOTALL)
content = re.sub(r"cursorDot\.style\.left = mouseX \+ 'px'; cursorDot\.style\.top = mouseY \+ 'px';", "", content)
content = re.sub(r"document\.querySelectorAll\('a, button, input, select, \.interactive'\)\.forEach\([^)]+\)\.addEventListener[^;]+;\s*\}\);", "", content, flags=re.DOTALL)
# Actually, the regex for removing the hover listener for cursor might fail if not exact, let's just do a string replace
cursor_hover_code = """        document.querySelectorAll('a, button, input, select, .interactive').forEach(el => {
            el.addEventListener('mouseenter', () => cursorDot.classList.add('hovering'));
            el.addEventListener('mouseleave', () => cursorDot.classList.remove('hovering'));
        });"""
content = content.replace(cursor_hover_code, '')

# 6. Change trail disappear time
content = content.replace('animation: fadeOut 0.8s forwards;', 'animation: fadeOut 0.5s forwards;')
content = content.replace('setTimeout(() => trail.remove(), 800);', 'setTimeout(() => trail.remove(), 500);')

# 7. Remove bouncing lights (Blobs)
content = re.sub(r'class Blob\s*\{.*?(?=function initDrips)', '', content, flags=re.DOTALL)
content = content.replace('blobs.forEach(b => { b.update(); b.draw(bgCtx); });', '')
content = content.replace('for (let i = 0; i < 10; i++) blobs.push(new Blob());', '')
content = content.replace('let blobs = []; ', '')

# 8. Box glow and hover
content = re.sub(
    r'\.how-card:hover, \.faq-card:hover\s*\{[^}]+\}',
    '.how-card:hover, .faq-card:hover {\n            box-shadow: 0 0 30px rgba(255, 107, 53, 0.4);\n            transform: translateY(-4px);\n            z-index: 10;\n        }',
    content
)

content = re.sub(
    r'\.dropzone:hover, \.dropzone\.dragover\s*\{[^}]+\}',
    '.dropzone:hover, .dropzone.dragover {\n            animation: none;\n            border-color: var(--accent-ember);\n            background: rgba(255, 107, 53, 0.05);\n            box-shadow: 0 0 30px rgba(255, 107, 53, 0.4);\n            transform: translateY(-4px);\n        }',
    content
)

content = content.replace('.art-panel { background-color: #000; position: relative; overflow: hidden; border: 1px solid var(--gap-color); }', '.art-panel { background-color: #000; position: relative; overflow: hidden; border: 1px solid var(--gap-color); transition: all 0.3s; }\n        .art-panel:hover { box-shadow: 0 0 30px rgba(255, 107, 53, 0.4); transform: translateY(-4px); z-index: 10; }')

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
