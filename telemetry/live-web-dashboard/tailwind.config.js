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
          text: "#ffffff",
          muted: "#ffffff",
          dim: "#ffffff",
          faint: "#ffffff"
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
        sans: ["system-ui", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "Cascadia Code", "Consolas", "monospace"]
      },
      fontSize: {
        "2xs": ["11px", "14px"]
      }
    }
  },
  plugins: []
};
