export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        hud: {
          bg: "#080b10",
          panel: "#0e131a",
          raise: "#141b24",
          line: "#1d2733",
          edge: "#27333f",
          text: "#e3eaf2",
          dim: "#8b9aab",
          faint: "#5c6a7a"
        },
        sig: {
          cyan: "#2dd4bf",
          blue: "#4c8dff",
          amber: "#f5a524",
          red: "#f2555a",
          green: "#3fcf8e",
          violet: "#9b8cff",
          pink: "#f472b6"
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"]
      },
      fontSize: {
        "2xs": ["10px", "12px"]
      }
    }
  },
  plugins: []
};
