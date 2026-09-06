/**
 * @fileoverview Reads the page's own words, without the furniture around them.
 *
 * The element list says what the agent can *do*. It never says what the page
 * *says* — and a task like "when does the archive open?" is answered by prose,
 * not by buttons. This script is the other half of reading a page: the article,
 * the search results, the prices, with the navigation and the footer left out.
 *
 * It matters most where there is no picture. A run against a model that cannot
 * see images has exactly two channels, and this is one of them.
 *
 * **Two ways this differs from element extraction, both deliberate.**
 *
 * 1. *The whole document, not the viewport.* Elements are offered a screenful
 *    at a time, because the agent can only click what is on screen. Text has no
 *    such limit — the answer to a question is as likely to be three screens
 *    down as in front of you, and scrolling to find it would cost a step per
 *    screen. So this reads the page top to bottom regardless of scroll.
 * 2. *Boilerplate is removed, not truncated.* A navigation bar is not a small
 *    amount of relevant text; it is a large amount of irrelevant text that
 *    repeats identically on every page of the site. Left in, it would crowd out
 *    the content under the token budget and make every page look alike.
 *
 * **Choosing where the content starts.** Three strategies, in order of how much
 * they are guessing:
 *
 * 1. `<main>`, or `role="main"` — the page telling you directly. Trusted.
 * 2. `<article>` — the page telling you where the content is, if not where the
 *    page's own body begins. Also trusted.
 * 3. The **largest text block**: score every container by how much non-link
 *    prose it holds, and take the winner. This is the guess, and it is the one
 *    that carries pages built with no semantic markup at all — which is most of
 *    them. Link text is discounted because a nav bar is mostly links, and
 *    without that discount a big enough menu wins against a short article.
 *
 * The chosen strategy is reported back, so a trace can say which one ran rather
 * than leaving a reader to wonder why a page read oddly.
 *
 * **Not traversed:** iframes and shadow roots, exactly as in element
 * extraction, and for the same reason — out of scope for v0, and consistently
 * so, since text the agent can read but never act on would be its own problem.
 */

(options) => {
  // Elements whose text is furniture wherever it appears. Removed outright
  // rather than scored down: there is no page on which the contents of a
  // <script> tag are the answer to anything.
  const BOILERPLATE_TAGS = new Set([
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "object",
    "video",
    "audio",
  ]);

  // The same idea expressed as ARIA landmarks, for pages that use <div> for
  // everything and say what it is with a role. `contentinfo` is the footer,
  // `banner` the masthead, `complementary` the sidebar.
  const BOILERPLATE_ROLES = new Set([
    "navigation",
    "banner",
    "contentinfo",
    "complementary",
    "menubar",
    "menu",
    "toolbar",
    "tablist",
  ]);

  // Elements after which a line break belongs. Without these the whole page
  // arrives as one paragraph, and a model reading it cannot tell a heading from
  // the sentence that follows it.
  const BLOCK_TAGS = new Set([
    "address",
    "article",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
  ]);

  // Containers considered when no <main> or <article> says where to look.
  const CANDIDATE_SELECTOR = "div, section, td, dd, li";

  // A container has to hold at least this many non-whitespace, non-link
  // characters to beat the
  // whole-body fallback -- roughly one sentence. The floor is low on purpose.
  // Its job is only to stop us confidently isolating a fragment on a page that
  // has no prose anywhere; it is not a judgement about how much content a page
  // ought to have. Set it high and a genuinely short page falls back to the
  // body, which drags in the very navigation this is trying to leave out.
  const MIN_BLOCK_CHARS = 40;

  /**
   * Whether an element's text is furniture rather than content.
   *
   * @param {Element} element The element to judge.
   * @returns {boolean} True if the element and its subtree should be skipped.
   */
  function isBoilerplate(element) {
    if (BOILERPLATE_TAGS.has(element.tagName.toLowerCase())) {
      return true;
    }
    const role = (element.getAttribute("role") || "").trim().toLowerCase();
    return role !== "" && BOILERPLATE_ROLES.has(role);
  }

  /**
   * Whether an element is hidden from sight, and so from the reader.
   *
   * Deliberately narrow. `display: none` and `visibility: hidden` are the two
   * ways a page says "this is not currently part of the document"; an element
   * merely scrolled off screen is not hidden, and its text still counts.
   *
   * @param {Element} element The element to test.
   * @returns {boolean} True if the element is not rendered at all.
   */
  function isHidden(element) {
    if (element.hasAttribute("hidden")) {
      return true;
    }
    if ((element.getAttribute("aria-hidden") || "") === "true") {
      return true;
    }
    const style = window.getComputedStyle(element);
    return style.display === "none" || style.visibility === "hidden";
  }

  /**
   * Whether this element ends a line of text.
   *
   * @param {Element} element The element to test.
   * @returns {boolean} True for block-level elements.
   */
  function isBlock(element) {
    return BLOCK_TAGS.has(element.tagName.toLowerCase());
  }

  /**
   * Gather the visible, non-furniture text under one node.
   *
   * Written as an explicit walk rather than `innerText` because `innerText`
   * has no way to leave the navigation out: it would have to be removed from
   * the live page first, and a read operation that edits the page it is reading
   * is a bug waiting for the first site that notices.
   *
   * @param {Node} node Where to start.
   * @param {string[]} out Collected fragments, appended to in place.
   * @param {boolean} skipLinks Leave anchor text out entirely. Used for
   *   scoring candidate containers, never for the extract itself -- an
   *   article's own links are part of what it says.
   * @returns {void}
   */
  function collect(node, out, skipLinks) {
    if (node.nodeType === Node.TEXT_NODE) {
      const value = node.nodeValue || "";
      if (value.trim() !== "") {
        out.push(value);
      }
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) {
      return;
    }
    const element = /** @type {Element} */ (node);
    if (isBoilerplate(element) || isHidden(element)) {
      return;
    }
    if (skipLinks && element.tagName.toLowerCase() === "a" && element.hasAttribute("href")) {
      return;
    }
    const block = isBlock(element);
    if (block) {
      out.push("\n");
    }
    for (const child of element.childNodes) {
      collect(child, out, skipLinks);
    }
    if (block) {
      out.push("\n");
    }
  }

  /**
   * Read one element's content text as a single tidy string.
   *
   * @param {Element} element The subtree root.
   * @param {boolean} [skipLinks=false] Leave anchor text out. For scoring only.
   * @returns {string} Text with runs of whitespace collapsed, blank lines
   *   reduced to one, and no leading or trailing space.
   */
  function textOf(element, skipLinks) {
    const parts = [];
    collect(element, parts, skipLinks === true);
    return parts
      .join("")
      // Every run of spaces and tabs becomes one space. Newlines are handled
      // separately below, so they are excluded from this class.
      .replace(/[^\S\n]+/g, " ")
      .replace(/ *\n */g, "\n")
      // Three or more newlines is a gap nobody needs; two is a paragraph break.
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  /**
   * Score a container by how much prose it holds, ignoring links entirely.
   *
   * A navigation bar and an article can hold a similar number of characters.
   * What separates them is that almost all of the navigation's characters sit
   * inside anchors, so leaving anchors out is what makes the comparison mean
   * something.
   *
   * Two details matter more than they look.
   *
   * Link text is *excluded from the walk* rather than subtracted afterwards.
   * Subtracting leaves the separators behind -- the line breaks around a menu
   * still count -- and those few characters are enough for a wrapper to
   * out-score the very child it wraps, which is the comparison this exists to
   * get right.
   *
   * Whitespace is not counted at all, for the same reason. A parent and a child
   * holding identical prose must score identically, or nesting resolves by an
   * accident of indentation.
   *
   * @param {Element} element The candidate container.
   * @returns {number} Non-whitespace characters of non-link text.
   */
  function proseScore(element) {
    return textOf(element, true).replace(/\s+/g, "").length;
  }

  /**
   * Find the element the page's content lives in.
   *
   * @returns {{node: Element, how: string}} The chosen root and the strategy
   *   that chose it, one of `main`, `article`, `largest_block` or `body`.
   */
  function pickRoot() {
    // Order matters and is not the order querySelector would give: an explicit
    // landmark outranks an <article>, even one that appears earlier.
    const main = document.querySelector("main") || document.querySelector("[role='main']");
    if (main && !isHidden(main)) {
      return { node: main, how: "main" };
    }
    const article = document.querySelector("article");
    if (article && !isHidden(article)) {
      return { node: article, how: "article" };
    }

    let best = null;
    let bestScore = MIN_BLOCK_CHARS;
    for (const candidate of document.body.querySelectorAll(CANDIDATE_SELECTOR)) {
      if (isBoilerplate(candidate) || isHidden(candidate)) {
        continue;
      }
      const score = proseScore(candidate);
      // Greater *or equal*, which resolves nesting towards the innermost
      // container. A wrapper <div> holding both the menu and the article scores
      // about the same as the article alone, because the menu is nearly all
      // link text and link text is discounted -- and the wrapper is reached
      // first, being the outer one. Taking the later of two equals therefore
      // picks the article and leaves the menu behind. Preferring the outer one
      // would keep the menu on every page built this way, which is most of them.
      if (score >= bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    if (best) {
      return { node: best, how: "largest_block" };
    }
    return { node: document.body, how: "body" };
  }

  const root = pickRoot();
  return {
    text: textOf(root.node),
    root: root.how,
    title: document.title || "",
  };
}
