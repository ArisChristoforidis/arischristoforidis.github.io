(() => {
  const form = document.querySelector("#visitor-console-form");
  const input = document.querySelector("#visitor-command");
  const output = document.querySelector("#visitor-console-output");
  const views = [...document.querySelectorAll("[data-visitor-view]")];

  if (!form || !input || !output || !views.length) return;

  const history = [];
  let historyIndex = 0;

  const promptText = "console@site:~$";

  const helpText = [
    "AVAILABLE COMMANDS",
    "  help              list available commands",
    "  traffic [range]   show the combined traffic report",
    "  pages             show the most visited paths",
    "  about             about this room",
    "  clear             clear the console output",
    "  exit              return to the main site",
  ].join("\n");

  const trafficHelpText = [
    "TRAFFIC PARAMETERS",
    "  traffic [range]   show a combined traffic report",
    "  traffic 30d       show the loaded 30-day report",
    "  traffic -h        show this parameter help",
    "",
    "Available data: 30d",
  ].join("\n");

  const showView = (name) => {
    const target = document.querySelector(`[data-visitor-view="${name}"]`);
    if (!target) return false;

    views.forEach((view) => {
      view.hidden = view !== target;
    });

    return true;
  };

  const hideViews = () => views.forEach((view) => { view.hidden = true; });

  const addLine = (command, response, responseClass = "") => {
    const commandLine = document.createElement("p");
    commandLine.className = "terminal-console-line";

    const prompt = document.createElement("span");
    prompt.textContent = promptText;
    commandLine.append(prompt, ` ${command}`);

    const responseLine = document.createElement("p");
    responseLine.className = `terminal-console-response ${responseClass}`.trim();
    responseLine.textContent = response;

    output.append(commandLine, responseLine);
    output.scrollTop = output.scrollHeight;
  };

  const runCommand = (rawCommand) => {
    const trimmed = rawCommand.trim();
    if (!trimmed) return;

    output.replaceChildren();
    hideViews();

    const parts = trimmed.split(/\s+/);
    const rawName = parts.shift().toLowerCase();
    const name = {
      "--help": "help",
      "-help": "help",
      "/help": "help",
      "?": "help",
    }[rawName] || rawName.replace(/^\//, "");
    const args = parts.map((part) => part.toLowerCase());

    if (name === "clear") {
      output.replaceChildren();
      hideViews();
      return;
    }

    if (name === "exit") {
      window.location.href = "/";
      return;
    }

    history.push(trimmed);
    historyIndex = history.length;

    switch (name) {
      case "help":
        addLine(trimmed, helpText);
        break;
      case "traffic":
        if (args.includes("-h") || args.includes("--help") || args[0] === "help") {
          addLine(trimmed, trafficHelpText);
          break;
        }
        showView("traffic");
        addLine(trimmed, args.length && args[0] !== "30d"
          ? `view mounted: combined traffic report (30d data; requested range ${args[0]} is not loaded yet)`
          : "view mounted: combined traffic report (30d)");
        break;
      case "pages":
        showView("pages");
        addLine(trimmed, "view mounted: most visited paths");
        break;
      case "about":
        addLine(trimmed, "Signals // christoforidis.dev\nA relay station for whatever the site is sending next.\nThe channel stays open.");
        break;
      default:
        addLine(trimmed, `command not found: ${rawName}\ntry '/help' for a list of available commands`, "terminal-console-response--error");
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runCommand(input.value);
    input.value = "";
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runCommand(input.value);
      input.value = "";
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      if (!history.length) return;
      historyIndex = Math.max(0, historyIndex - 1);
      input.value = history[historyIndex];
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!history.length) return;
      historyIndex = Math.min(history.length, historyIndex + 1);
      input.value = history[historyIndex] || "";
    }
  });
})();
