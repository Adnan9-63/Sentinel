/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base:     '#f4f5f7',   // Light mode base background
        surface:  '#ffffff',   // Cards
        surface2: '#f8f9fa',   // Card hover / secondary surface
        hairline: '#ebecf0',   // Crisp borders
        ink:      '#172b4d',   // Primary text
        muted:    '#5e6c84',   // Secondary text
        brand:    '#0D94FB',   // Dodger Blue - primary action
        deep:     '#012652',   // Prussian Blue - headers
        allow:    '#04db7c',   // Success green
        review:   '#ff991f',   // Warning amber
        flagged:  '#de350b',   // Danger red
        dim:      '#091e420a', // Subtle shadow/dim
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['"Fira Code"', 'monospace'],
      },
      borderRadius: {
        sm: '4px',
        md: '6px',
        lg: '12px',
      },
      boxShadow: {
        'card': '0 1px 1px rgba(9, 30, 66, 0.25), 0 0 1px rgba(9, 30, 66, 0.31)',
        'float': '0 4px 8px -2px rgba(9, 30, 66, 0.25), 0 0 1px rgba(9, 30, 66, 0.31)',
      }
    },
  },
  plugins: [],
}
