(() => {
  const grid = document.querySelector("[data-messier-grid]");
  const detail = document.querySelector("[data-messier-detail]");

  if (!grid || !detail) return;

  const typeLabels = {
    SN: "supernova remnant",
    OC: "open cluster",
    GC: "globular cluster",
    PN: "planetary nebula",
    DN: "emission nebula",
    AS: "asterism",
    DS: "double star",
    MW: "star cloud",
    SG: "spiral galaxy",
    BG: "barred galaxy",
    LG: "lenticular galaxy",
    EG: "elliptical galaxy",
    IG: "irregular galaxy",
  };
  const nameAliases = {
    42: "Orion Nebula",
  };
  const monthLabels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const monthNames = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ];
  const fallbackProject = {
    camera: "Sony a6700",
    lens: "Sigma 18-50mm",
    captured: {},
    visibilityOverrides: {},
  };

  const parseRaHours = (rightAscension) => {
    const match = String(rightAscension || "").match(/(\d{1,2})h\s*([\d.]+)?/i);
    if (!match) return 0;
    return Number(match[1]) + (Number(match[2]) || 0) / 60;
  };

  const normalizedMonth = (month) => ((month - 1) % 12 + 12) % 12 + 1;

  const windowMonthsFor = (record, project) => {
    const override = project.visibilityOverrides?.[String(record.number)];
    if (override?.length) return override;

    // Approximate the months when the target is highest during the dark hours.
    // The RA-to-month mapping is intentionally simple and readable for a field log.
    const culminationMonth = normalizedMonth(Math.round(parseRaHours(record.ra) / 2 + 9));
    return Array.from({ length: 4 }, (_, index) => normalizedMonth(culminationMonth + index));
  };

  const formatWindow = (months) => {
    if (!months?.length) return "Visibility window to be added";
    const first = months[0];
    const last = months[months.length - 1];
    return months.length === 1
      ? monthNames[first - 1]
      : `${monthNames[first - 1]} - ${monthNames[last - 1]}`;
  };

  const formatWindowShort = (months) => {
    if (!months?.length) return "Visibility window to be added";
    const first = months[0];
    const last = months[months.length - 1];
    return months.length === 1
      ? monthLabels[first - 1]
      : `${monthLabels[first - 1]} - ${monthLabels[last - 1]}`;
  };

  const formatDates = (dates) => {
    if (!dates?.length) return "Not captured yet";
    return dates.map((date) => {
      const [year, month, day] = date.split("-").map(Number);
      return new Intl.DateTimeFormat("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(Date.UTC(year, month - 1, day)));
    }).join(" + ");
  };

  const formatCatalogDetails = (record) => {
    const details = [];
    if (record.ra && record.dec) details.push(`RA ${record.ra} · Dec ${record.dec}`);
    if (record.magnitude) details.push(`mag ${record.magnitude}`);
    return details.join(" · ");
  };

  const monthSegments = (months) => {
    const active = new Set(months);
    const segments = [];
    let month = 1;

    while (month <= 12) {
      if (!active.has(month)) {
        month += 1;
        continue;
      }

      const start = month;
      while (month <= 12 && active.has(month)) month += 1;
      segments.push({
        start: (start - 1) / 12 * 100,
        width: (month - start) / 12 * 100,
      });
    }

    return segments;
  };

  const getInfo = (record, project) => {
    const capture = project.captured?.[String(record.number)] || null;
    const windowMonths = windowMonthsFor(record, project);
    const displayName = nameAliases[record.number] || record.name ||
      (record.ngc ? `NGC ${record.ngc}` : `Messier object ${record.number}`);

    return {
      ...record,
      name: displayName,
      meta: typeLabels[record.type] || "catalog target",
      wikipedia: `https://en.wikipedia.org/wiki/Messier_${record.number}`,
      windowMonths,
      window: formatWindow(windowMonths),
      state: capture ? "captured" : "planned",
      dates: formatDates(capture?.dates),
      images: capture?.images || [],
      thumbnail: capture?.thumbnail || null,
      fieldNote: capture?.fieldNote || null,
    };
  };

  const isVisibleNow = (info) => info.state !== "captured" && info.windowMonths.includes(new Date().getMonth() + 1);

  const updateProgress = (capturedCount, total) => {
    const percentage = capturedCount / total * 100;
    const progressFill = document.querySelector("[data-progress-fill]");
    const progressPercent = document.querySelector("[data-progress-percent]");
    const progressCount = document.querySelector("[data-progress-count]");
    const progressCaptured = document.querySelector("[data-progress-captured]");
    const progressAhead = document.querySelector("[data-progress-ahead]");
    const progressBar = document.querySelector('[role="progressbar"]');

    if (progressFill) progressFill.style.setProperty("--progress", `${percentage}%`);
    if (progressPercent) progressPercent.textContent = `${Math.round(percentage)}% captured`;
    if (progressCount) progressCount.textContent = String(capturedCount).padStart(2, "0");
    if (progressCaptured) progressCaptured.textContent = `${capturedCount} photographed`;
    if (progressAhead) progressAhead.textContent = `${total - capturedCount} still ahead`;
    if (progressBar) progressBar.setAttribute("aria-valuenow", String(capturedCount));
  };

  const loadCatalog = async () => {
    const response = await fetch("/assets/astro/messier.json");
    if (!response.ok) throw new Error(`Messier catalog request failed: ${response.status}`);
    return response.json();
  };

  const render = ({ catalog, project = fallbackProject }) => {
    const records = catalog
      .map((record) => ({
        number: Number(String(record.M).replace("M", "")),
        name: record.N || "",
        ngc: record.NGC || "",
        type: record.T || "",
        magnitude: record.V || "",
        size: record.S || "",
        ra: record.RA || "",
        dec: record.Dec || "",
        constellation: record.Con || "",
      }))
      .sort((a, b) => a.number - b.number);
    const infoByNumber = new Map(records.map((record) => [record.number, getInfo(record, project)]));
    const stateLabels = {
      captured: "captured",
      planned: "still on the horizon",
    };
    const capturedCount = records.filter((record) => infoByNumber.get(record.number).state === "captured").length;

    const currentMonth = new Date().getMonth() + 1;
    const upcoming = [...infoByNumber.values()]
      .filter((info) => info.state !== "captured")
      .filter((info) => !info.windowMonths.includes(currentMonth))
      .map((info) => ({
        info,
        monthsUntilStart: (info.windowMonths[0] - currentMonth + 12) % 12 || 12,
      }))
      .filter((item) => item.monthsUntilStart > 0)
      .sort((a, b) => a.monthsUntilStart - b.monthsUntilStart || Number(a.info.magnitude) - Number(b.info.magnitude))
      .slice(0, 4);
    const windowRows = document.querySelector("[data-window-rows]");

    if (windowRows) {
      windowRows.replaceChildren(...upcoming.map(({ info }) => {
        const row = document.createElement("div");
        const target = document.createElement("div");
        const targetId = document.createElement("strong");
        const targetName = document.createElement("span");
        const track = document.createElement("div");
        const meta = document.createElement("span");

        row.className = "messier-window-row";
        target.className = "messier-window-target";
        targetId.textContent = `M${info.number}`;
        targetName.textContent = info.name;
        target.append(targetId, targetName);
        track.className = "messier-window-track";
        track.setAttribute("aria-label", `M${info.number} visibility window`);
        track.append(...monthSegments(info.windowMonths).map((segment) => {
          const range = document.createElement("span");
          range.className = "messier-window-range";
          range.style.setProperty("--window-start", `${segment.start}%`);
          range.style.setProperty("--window-width", `${segment.width}%`);
          return range;
        }));
        meta.className = "messier-window-meta";
        meta.textContent = formatWindowShort(info.windowMonths);
        row.append(target, track, meta);
        return row;
      }));
    }

    const showObject = (number) => {
      const info = infoByNumber.get(number) || getInfo({ number }, project);
      const state = info.state;
      const fieldNoteLink = detail.querySelector("[data-detail-link]");
      const wikipediaLink = detail.querySelector("[data-detail-wikipedia]");
      const gallery = detail.querySelector("[data-detail-gallery]");
      const galleryItems = detail.querySelector("[data-detail-gallery-items]");

      detail.querySelector("[data-detail-state]").textContent = stateLabels[state];
      detail.querySelector("[data-detail-title]").textContent = info.name;
      detail.querySelector("[data-detail-meta]").textContent = `M${number} / ${info.meta}`;
      detail.querySelector("[data-detail-catalog]").textContent = formatCatalogDetails(info);
      detail.querySelector("[data-detail-window]").textContent = info.window;
      wikipediaLink.href = info.wikipedia;
      galleryItems.replaceChildren();
      info.images.forEach((image) => {
        const link = document.createElement("a");
        const thumbnail = document.createElement("img");
        link.className = "messier-detail-thumb";
        link.href = image.href || image.src;
        link.target = "_blank";
        link.rel = "noreferrer";
        thumbnail.src = image.src;
        thumbnail.alt = image.alt || "";
        thumbnail.loading = "lazy";
        thumbnail.dataset.imageCaption = image.caption || "";
        thumbnail.style.objectPosition = image.position || "center";
        link.append(thumbnail);
        galleryItems.append(link);
      });
      gallery.hidden = !info.images.length;

      if (info.fieldNote) {
        fieldNoteLink.hidden = false;
        fieldNoteLink.href = info.fieldNote;
      } else {
        fieldNoteLink.hidden = true;
        fieldNoteLink.removeAttribute("href");
      }

      grid.querySelectorAll(".messier-cell").forEach((cell) => {
        cell.dataset.selected = cell.dataset.number === String(number) ? "true" : "false";
      });
    };

    updateProgress(capturedCount, records.length);
    grid.replaceChildren();
    grid.setAttribute("aria-busy", "false");

    records.forEach((record) => {
      const info = infoByNumber.get(record.number);
      const visibleNow = isVisibleNow(info);
      const cell = document.createElement("button");
      const thumbnail = info.thumbnail || info.images[0];

      cell.className = "messier-cell";
      cell.type = "button";
      cell.dataset.number = record.number;
      cell.dataset.state = info.state;
      cell.dataset.visible = visibleNow ? "true" : "false";
      cell.dataset.selected = "false";
      cell.textContent = record.number;
      if (thumbnail) {
        cell.classList.add("messier-cell--image");
        cell.style.backgroundImage = `linear-gradient(rgba(24, 42, 62, 0.2), rgba(24, 42, 62, 0.34)), url("${thumbnail.src}")`;
        cell.style.backgroundPosition = thumbnail.position || "center";
      }
      cell.setAttribute("aria-label", `M${record.number}, ${stateLabels[info.state]}${visibleNow ? ", visible now" : ""}`);
      cell.title = `M${record.number} - ${visibleNow ? "visible now" : stateLabels[info.state]}`;
      cell.addEventListener("click", () => showObject(record.number));
      grid.append(cell);
    });

    const defaultNumber = infoByNumber.has(31) ? 31 : records[0].number;
    showObject(defaultNumber);
  };

  loadCatalog()
    .then(render)
    .catch((error) => {
      console.error("Unable to load the Messier catalog", error);
      grid.setAttribute("aria-busy", "false");
    });
})();
