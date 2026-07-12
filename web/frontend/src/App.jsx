import { useState } from "react";
import Header from "./components/Header.jsx";
import ImageMode from "./pages/ImageMode.jsx";
import VideoMode from "./pages/VideoMode.jsx";

const MODES = [
  { id: "image", label: "Chế độ ảnh" },
  { id: "video", label: "Chế độ video" }
];

// Top-level UI shell with mode switching.
export default function App() {
  const [mode, setMode] = useState("image");

  return (
    <div className="min-h-screen text-ink">
      <Header modes={MODES} mode={mode} onModeChange={setMode} />
      <main className="mx-auto max-w-7xl px-6 py-2">
        {mode === "image" ? <ImageMode /> : <VideoMode />}
      </main>
    </div>
  );
}
