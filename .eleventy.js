module.exports = async function(eleventyConfig) {
  // Use dynamic import to load Eleventy plugins
  const { EleventyHtmlBasePlugin } = await import("@11ty/eleventy");
  const { DateTime } = await import("luxon");
  const markdownItModule = await import("markdown-it");
  const markdownItAttrsModule = await import("markdown-it-attrs");

  const markdownIt = markdownItModule.default;
  const markdownItAttrs = markdownItAttrsModule.default;

  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/assets");

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

  const md = markdownIt({ html: true }).use(markdownItAttrs);
  eleventyConfig.setLibrary("md", md);

  return {
    dir: {
      input: "src",
      output: "_site",
    },
  };
};
