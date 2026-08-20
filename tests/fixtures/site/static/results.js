/**
 * @fileoverview Renders the catalogue search results page.
 *
 * The catalogue is a fixed, ordered list held in this file. Nothing is random
 * and nothing is fetched, so the same query always produces the same page —
 * which is what makes the result count usable as a target a test can assert
 * on. The query arrives in the URL, because a static file server cannot run
 * any code of its own.
 */

/**
 * One catalogue record.
 * @typedef {{title: string, href: string, tags: !Array<string>}} Record
 */

/**
 * The whole catalogue, in display order.
 * @const {!Array<Record>}
 */
const CATALOGUE = [
  {
    title: "Membership application",
    href: "form-step-1.html",
    tags: ["membership", "join", "form"],
  },
  {
    title: "Reading room refurbishment, 2026",
    href: "article.html",
    tags: ["reading room", "building", "news"],
  },
  {
    title: "Data and privacy notice for members",
    href: "consent.html",
    tags: ["privacy", "membership", "notice"],
  },
  {
    title: "Using the digital archive",
    href: "article.html#archive",
    tags: ["archive", "reading room"],
  },
  {
    title: "Interlibrary loan request",
    href: "form-step-1.html",
    tags: ["loan", "form", "membership"],
  },
  {
    title: "Catalogue search tips",
    href: "search.html",
    tags: ["search", "help"],
  },
];

/**
 * Tests whether a record matches a search query.
 *
 * Matching is a case-insensitive substring test across the title and the
 * subject tags. An empty query matches everything, so a bare visit to the
 * results page lists the full catalogue.
 *
 * @param {Record} record The catalogue record to test.
 * @param {string} query Lower-cased search terms, possibly empty.
 * @return {boolean} True if the record should appear in the results.
 */
function matches(record, query) {
  if (!query) {
    return true;
  }
  const haystack = (record.title + " " + record.tags.join(" ")).toLowerCase();
  return haystack.includes(query);
}

/**
 * Reads the query from the URL and fills the page in from the catalogue.
 *
 * Writes the result list, the count line, the visibility of the empty-result
 * message, and the value of the refine box.
 *
 * @return {void}
 */
function render() {
  const query = (new URLSearchParams(window.location.search).get("q") || "")
    .trim()
    .toLowerCase();
  const hits = CATALOGUE.filter((record) => matches(record, query));

  const list = document.getElementById("results-list");
  for (const record of hits) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = record.href;
    link.textContent = record.title;
    const tags = document.createElement("div");
    tags.className = "tags";
    tags.textContent = "Subjects: " + record.tags.join(", ");
    item.append(link, tags);
    list.append(item);
  }

  // The count line is the page's summary of itself, so it is phrased to read
  // naturally whether there are none, one, or many.
  const noun = hits.length === 1 ? "record" : "records";
  const count = document.getElementById("result-count");
  count.textContent = query
    ? `${hits.length} ${noun} for "${query}"`
    : `${hits.length} ${noun} in the catalogue`;

  document.getElementById("no-results").hidden = hits.length > 0;
  // Carrying the query into the refine box mirrors how real search pages
  // behave, and lets a second search build on the first.
  document.getElementById("refine-query").value = query;
}

render();
