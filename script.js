async function loadResults() {
  try {
    // Fetch the CSV file
    const response = await fetch("results.csv");
    const csvText = await response.text();

    // Parse CSV (skip header row)
    let header = csvText.split("\n")[0];
    header = header.replace(/_/g, " ");
    header = header.replace(/\b\w/g, (char) => char.toUpperCase());
    const rows = csvText
      .split("\n")
      .slice(1)
      .filter((row) => row.trim() !== "");

    // Create table HTML
    const table = `
            <table>
                <thead>
                    <tr>
                        ${header
                          .split(",")
                          .map((col) => `<th>${col}</th>`)
                          .join("")}
                    </tr>
                </thead>
                <tbody>
                    ${rows
                      .map(
                        (row) =>
                          `<tr>${row
                            .split(",")
                            .map((col) => `<td>${col}</td>`)
                            .join("")}</tr>`,
                      )
                      .join("")}
                </tbody>
            </table>
        `;

    // Add table to content div
    document.getElementById("content").innerHTML = table;
  } catch (error) {
    console.error("Error loading results:", error);
    document.getElementById("content").innerHTML =
      "<p>Error loading results. Please try again later.</p>";
  }
}

// Load results when page loads
document.addEventListener("DOMContentLoaded", loadResults);
