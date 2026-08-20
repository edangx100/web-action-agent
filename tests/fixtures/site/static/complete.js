/**
 * @fileoverview Echoes a submitted membership application back to the reader.
 *
 * The confirmation page is the end of the multi-step form, and reading it back
 * is how a test confirms that every answer survived the journey rather than
 * only the last one.
 */

/**
 * The fields to display, as [query parameter, visible label, element id].
 *
 * The order is fixed so that identical input always renders an identical page.
 *
 * @const {!Array<!Array<string>>}
 */
const FIELDS = [
  ["name", "Full name", "summary-name"],
  ["email", "Email address", "summary-email"],
  ["branch", "Home branch", "summary-branch"],
];

const params = new URLSearchParams(window.location.search);
const summary = document.getElementById("submission-summary");

for (const [param, label, id] of FIELDS) {
  const term = document.createElement("dt");
  term.textContent = label;
  const value = document.createElement("dd");
  value.id = id;
  // A missing answer is stated plainly rather than left blank, so a form
  // filled in incompletely is visibly different from one filled in correctly.
  value.textContent = params.get(param) || "not provided";
  summary.append(term, value);
}
