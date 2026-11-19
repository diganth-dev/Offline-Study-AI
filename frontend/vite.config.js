import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      includeAssets: [
        'favicon.svg',
        'apple-touch-icon.png',
        'icons/*.png'
      ],
      manifest: {
        name: 'Offline Study AI',
        short_name: 'Study AI',
        description: 'Offline, private AI study assistant for college students. Upload notes, ask questions, and review answers with exact citations.',
        theme_color: '#0f172a',
        background_color: '#f8fafc',
        display: 'standalone',
        orientation: 'any',
        start_url: '/',
        scope: '/',
        id: '/',
        icons: [
          {
            src: '/icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: '/icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: '/icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any'
          },
          {
            src: '/icons/maskable-icon-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'maskable'
          },
          {
            src: '/icons/maskable-icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable'
          }
        ],
        categories: ['education', 'productivity', 'utilities'],
        shortcuts: [
          {
            name: 'AI Study Chat',
            short_name: 'Chat',
            description: 'Ask questions based on your study materials',
            url: '/?tab=chat',
            icons: [{ src: '/icons/icon-192x192.png', sizes: '192x192' }]
          },
          {
            name: 'AI Summarizer',
            short_name: 'Summarize',
            description: 'Generate chapter summaries and key takeaways',
            url: '/?tab=summarizer',
            icons: [{ src: '/icons/icon-192x192.png', sizes: '192x192' }]
          },
          {
            name: 'Important Questions',
            short_name: 'Questions',
            description: 'Generate exam questions from documents',
            url: '/?tab=questions',
            icons: [{ src: '/icons/icon-192x192.png', sizes: '192x192' }]
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [
          /^\/api\//,
          /^\/health/,
          /^\/documents/,
          /^\/index-pdf/,
          /^\/ask/,
          /^\/search/,
          /^\/summarize/,
          /^\/important-questions/
        ],
        runtimeCaching: [
          {
            // Specifically exclude backend API routes from being handled or cached by service worker
            urlPattern: ({ url }) => {
              return (
                url.port === '8000' ||
                url.pathname.startsWith('/api') ||
                url.pathname.startsWith('/health') ||
                url.pathname.startsWith('/documents') ||
                url.pathname.startsWith('/index-pdf') ||
                url.pathname.startsWith('/ask') ||
                url.pathname.startsWith('/search') ||
                url.pathname.startsWith('/summarize') ||
                url.pathname.startsWith('/important-questions')
              );
            },
            handler: 'NetworkOnly'
          }
        ]
      }
    })
  ]
});
