import { useState } from "react";
import Header from "./components/Header.jsx";
import ImageMode from "./pages/ImageMode.jsx";
import VideoMode from "./pages/VideoMode.jsx";

const MODES = [
  { id: "image", label: "Image Mode" },
  { id: "video", label: "Video Mode" }
];

// Top-level UI shell with mode switching.
export default function App() {
  const [mode, setMode] = useState("image");

  return (
    <div className="min-h-screen text-ink">
      <Header modes={MODES} mode={mode} onModeChange={setMode} />
      <main className="mx-auto max-w-7xl px-6 pb-10">
        {mode === "image" ? <ImageMode /> : <VideoMode />}
      </main>
    </div>
  );
}
