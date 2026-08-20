module.exports = async function(eleventyConfig) {
  // Use dynamic import to load Eleventy plugins
  const { EleventyHtmlBasePlugin } = await import("@11ty/eleventy");
  const { DateTime } = await import("luxon");
  const markdownItModule = await import("markdown-it");
  const markdownItAttrsModule = await import("markdown-it-attrs");
  const { createHighlighter } = await import("shiki");

  const markdownIt = markdownItModule.default;
  const markdownItAttrs = markdownItAttrsModule.default;

  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");
  eleventyConfig.addPassthroughCopy("src/assets");

  eleventyConfig.addGlobalData("buildYear", new Date().getFullYear());

  eleventyConfig.addPreprocessor("skip-drafts", "md", (data) => {
    if (data.draft === true) {
      return false;
    }
  });

  eleventyConfig.addCollection("posts", (collectionApi) => {
    return collectionApi
      .getFilteredByTag("posts")
      .filter((item) => item.data.draft !== true);
  });

  eleventyConfig.addCollection("programming", (collectionApi) => {
    return collectionApi
      .getFilteredByTag("programming")
      .filter((item) => item.data.draft !== true);
  });

  eleventyConfig.addPlugin(EleventyHtmlBasePlugin);

  eleventyConfig.addFilter("htmlDateString", (dateObj) => {
    return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat("yyyy-LL-dd");
  });

  eleventyConfig.addFilter("readableDate", (dateObj) => {
    return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat("dd LLL yyyy");
  });

  eleventyConfig.addFilter("truncate", function (text, length, killwords, end) {
    if (text && text.length > length) {
      text = text.substring(0, length);
      if (killwords) {
        text = text.substring(0, text.lastIndexOf(" "));
      }
      text += end != null ? end : "...";
    }
    return text;
  });

  const codeHighlighter = await createHighlighter({
    themes: ["github-light", "github-dark"],
    langs: [
      "bash",
      "css",
      "html",
      "javascript",
      "json",
      "markdown",
      "python",
      "text",
      "typescript",
    ],
  });

  const codeLanguages = new Set([
    "bash",
    "css",
    "html",
    "javascript",
    "js",
    "json",
    "markdown",
    "md",
    "python",
    "py",
    "text",
    "typescript",
    "ts",
  ]);

  const md = markdownIt({
    html: true,
    highlight(code, languageInfo) {
      const requestedLanguage = (languageInfo || "text").split(/\s+/)[0].toLowerCase();
      const language = codeLanguages.has(requestedLanguage)
        ? requestedLanguage
        : "text";
      const shikiLanguage = language === "js"
        ? "javascript"
        : language === "md"
          ? "markdown"
          : language === "py"
            ? "python"
            : language === "ts"
              ? "typescript"
              : language;
      const highlighted = codeHighlighter.codeToHtml(code, {
        lang: shikiLanguage,
        themes: {
          light: "github-light",
          dark: "github-dark",
        },
      });

      return highlighted.replace(
        '<pre class="shiki',
        '<pre data-language="' + language + '" class="shiki',
      );
    },
  }).use(markdownItAttrs);
  eleventyConfig.setLibrary("md", md);

  return {
    dir: {
      input: "src",
      output: "_site",
    },
  };
};
