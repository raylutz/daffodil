import re
def on_page_content(html, **kw):
    return re.sub(r'<span class="o">&lt;</span><span class="n">BLANKLINE</span><span class="o">&gt;</span>\n?', '', html)
