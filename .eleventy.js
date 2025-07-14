const { EleventyHtmlBasePlugin } = require("@11ty/eleventy");
const { DateTime } = require("luxon");
const markdownIt = require("markdown-it");
const markdownItAttrs = require("markdown-it-attrs");

module.exports = function (eleventyConfig) {
    eleventyConfig.addPassthroughCopy("src/css");
    eleventyConfig.addPassthroughCopy("src/assets");

    // Add a plugin to handle base paths (useful for subdirectories on servers)
    eleventyConfig.addPlugin(EleventyHtmlBasePlugin);

    // Add a filter to format dates for HTML datetime attribute
    eleventyConfig.addFilter("htmlDateString", (dateObj) => {
        return DateTime.fromJSDate(dateObj, { zone: 'utc' }).toFormat('yyyy-LL-dd');
    });

    // Add a filter to format dates for human readability
    eleventyConfig.addFilter("readableDate", (dateObj) => {
        return DateTime.fromJSDate(dateObj, { zone: 'utc' }).toFormat('dd LLL yyyy');
    });

    eleventyConfig.addFilter("truncate", function (text, length, killwords, end) {
        if (text && text.length > length) { // Added null/undefined check for 'text'
            text = text.substring(0, length);
            if (killwords) {
                text = text.substring(0, text.lastIndexOf(' '));
            }
            text += (end != null) ? end : '...';
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