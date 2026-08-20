/* The catalogue is a fixed, ordered list. Nothing here is random and nothing is
   fetched: the same query always renders the same page, which is what makes a
   result count usable as a verification target in a test. */

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

function matches(record, query) {
  if (!query) {
    return true;
  }
  const haystack = (record.title + " " + record.tags.join(" ")).toLowerCase();
  return haystack.includes(query);
}

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

  const noun = hits.length === 1 ? "record" : "records";
  const count = document.getElementById("result-count");
  count.textContent = query
    ? `${hits.length} ${noun} for "${query}"`
    : `${hits.length} ${noun} in the catalogue`;

  document.getElementById("no-results").hidden = hits.length > 0;
  document.getElementById("refine-query").value = query;
}

render();
