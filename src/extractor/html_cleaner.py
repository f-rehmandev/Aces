from bs4 import BeautifulSoup, Comment


def clean_html(raw_html: str) -> str:
    """
    Strips noise from raw HTML (scripts, styles, nav, ads, comments, tracking
    attributes) so an LLM sees mostly the meaningful content — critical for
    fitting large real-world pages (Amazon, Daraz) into a prompt.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove tags that never contain product/content data
    NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Strip attributes that are pure noise for extraction (styling, tracking, event handlers)
    KEEP_ATTRS = {"href", "src", "alt", "title"}
    for tag in soup.find_all(True):
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr not in KEEP_ATTRS:
                del tag.attrs[attr]

    return str(soup)


if __name__ == "__main__":
    sample = """
    <html>
      <head><style>.x{color:red}</style></head>
      <body>
        <nav>Home | About</nav>
        <script>trackUser();</script>
        <div class="product" onclick="buy()" data-track="xyz">
            <h2 style="color:blue;">Wireless Mouse</h2>
            <span>$24.99</span>
        </div>
        <!-- an ad slot -->
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    cleaned = clean_html(sample)
    print("Original length:", len(sample))
    print("Cleaned length:", len(cleaned))
    print("\nCleaned HTML:\n")
    print(cleaned)