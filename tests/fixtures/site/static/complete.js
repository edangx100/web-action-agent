/* Echoes the submitted application back. The field order is fixed so the
   rendered summary is identical for identical input. */

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
  value.textContent = params.get(param) || "not provided";
  summary.append(term, value);
}
