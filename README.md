# christoforidis.dev

Personal website built with [Eleventy](https://www.11ty.dev/), Markdown, and plain CSS/JavaScript.

## Local development

The first time you work on the site:

```powershell
npm install
```

Start a local server with live reload:

```powershell
npm run dev
```

When you are ready to verify the production build:

```powershell
npm run check
```

The build cleans and recreates `_site/`. It is ignored by Git and should not be edited directly.

## Publishing

Push to the `master` branch. GitHub Actions builds the site and publishes `_site` to GitHub Pages.

Pull requests also run the build check, but do not deploy.

## Writing a post

On Windows, create a draft with:

```powershell
npm run new:post -- "A title for the new post"
```

The new file will be created under `src/posts/`. Set `draft: false` when it is ready to publish, then run `npm run dev` and preview it locally.

Projects can use the same front matter shape as posts, with `layout: "prog.njk"` and a `categories` list.
