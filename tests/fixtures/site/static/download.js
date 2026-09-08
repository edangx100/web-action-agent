/**
 * @fileoverview Offers a small file for download, built inside the browser.
 *
 * A `download` attribute on a link pointing at a blob is the shortest way to
 * make a browser offer a file, and it needs nothing of the server -- which
 * keeps the fixture site a folder of static files. The page's status line is
 * updated too, so a test can tell the click was received even though the
 * download itself is deliberately ignored.
 */

const rows = [
  "bundle,title,year",
  "B-001,Trustees' minutes,1908",
  "B-002,Reading room plans,1911",
  "B-003,Bequest correspondence,1913",
].join("\n");

document.getElementById("download-button").addEventListener("click", () => {
  const file = new Blob([rows], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(file);
  link.download = "ledger-extract.csv";
  link.click();
  URL.revokeObjectURL(link.href);
  document.getElementById("download-status").textContent = "Export offered.";
});
