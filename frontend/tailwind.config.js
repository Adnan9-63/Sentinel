/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base: '#0E1420',
        surface: '#161D2C',
        surface2: '#1D2740',
        hairline: '#2A3448',
        ink: '#E8ECF4',
        muted: '#8891A6',
        teal: '#4FD1C5',
        allow: '#6FCF97',
        review: '#F2B84B',
        flagged: '#E8637A',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
