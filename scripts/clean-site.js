const fs = require("node:fs");
const path = require("node:path");

const outputDirectory = path.join(__dirname, "..", "_site");
fs.rmSync(outputDirectory, { recursive: true, force: true });
console.log(`Cleaned ${outputDirectory}`);
