// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Benchmarking Attacks on Learning with Errors (LWE)',
  tagline: 'Measuring concrete attack performance against LWE-based cryptosystems',
  favicon: 'img/chart.svg',

  // Set the production url of your site here
  url: 'https://facebookresearch.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/LWE-benchmarking/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebookresearch', // Usually your GitHub org/user name.
  projectName: 'LWE-benchmarking', // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'LWE Attack Benchmark',
        logo: {
          alt: 'Bar Chart Logo',
          src: 'img/chart.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            to: 'benchmark', label: 'Benchmark', position: 'left'
          },
          {
            to: 'about-us', label: 'About Us', position: 'left'
          },
          {
            href: 'https://eprint.iacr.org/2024/1229',
            label: 'Paper',
            position: 'right',
          },
          {
            href: 'https://github.com/facebookresearch/LWE-benchmarking',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
              {
                label: 'Contribute',
                to: '/docs/contribute'
              },
              {
                label: 'Benchmark',
                to: 'benchmark'
              }
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebookresearch/LWE-benchmarking',
              },
              {
                label: 'Paper',
                href: 'https://eprint.iacr.org/2024/1229',
              },
              {
                label: 'About Us',
                to: 'about-us',
              }
            ],
          },
          {
            title: 'Legal',
            // Please do not remove the privacy and terms, it's a legal requirement.
            items: [
              {
                label: 'Privacy',
                href: 'https://opensource.facebook.com/legal/privacy/',
                target: '_blank',
                rel: 'noreferrer noopener',
              },
              {
                label: 'Terms',
                href: 'https://opensource.facebook.com/legal/terms/',
                target: '_blank',
                rel: 'noreferrer noopener',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. and affiliates. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
