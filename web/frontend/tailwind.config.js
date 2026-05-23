/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B1220",
        mist: "#F5F7FC",
        accent: "#2563EB",
        accentSoft: "#DCE7FF",
      },
      boxShadow: {
        soft: "0 12px 40px rgba(21, 38, 80, 0.12)",
        insetSoft: "inset 0 1px 0 rgba(255,255,255,0.4)",
      },
    },
  },
  plugins: [],
};
