# Introduction à Python

This repository contains the course materials for the Applied Python undergraduate course, built with MyST.

## Building the Website Locally

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn

### Installation

1. Install MyST CLI globally:

```bash
npm install -g mystmd
```

2. Install Python dependencies (optional, for Jupyter notebook support):

```bash
pip install -r requirements.txt
```

### Build and Preview

1. Build the website:

```bash
myst build
```

2. Start a local development server:

```bash
myst start
```

The website will be available at `http://localhost:3000`

## Deploying to GitHub Pages

### Option 1: Using GitHub Actions (Recommended)

1. Push your code to GitHub
2. The GitHub Action will automatically build and deploy to GitHub Pages
3. Enable GitHub Pages in your repository settings (Settings > Pages > Source: gh-pages branch)

### Option 2: Manual Deployment

1. Build the site:

```bash
myst build --html
```

2. The built site will be in the `_build/html` directory

3. Deploy to GitHub Pages:

```bash
myst build --gh-pages
```

## Project Structure

```
.
├── myst.yml              # MyST configuration
├── index.md              # Homepage
├── lectures/             # Lecture markdown files
│   ├── lecture1.md
│   └── lecture2.md
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Adding New Lectures

1. Create a new markdown file in the `lectures/` directory
2. Add the lecture to `myst.yml` under `site.nav`
3. Rebuild the site with `myst build`

## Resources

- [MyST Documentation](https://mystmd.org/)
- [MyST Markdown Guide](https://mystmd.org/guide)
