export const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp";
export const VIDEO_ACCEPT =
  "video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm";
export const MAX_VIDEO_UPLOAD_MB =
  Number(import.meta.env.VITE_MAX_VIDEO_UPLOAD_MB) || 500;

const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"];

const VIDEO_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
  "video/webm",
];
const VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

function getFileExt(fileName) {
  const dotIndex = fileName.lastIndexOf(".");
  if (dotIndex < 0) {
    return "";
  }
  return fileName.toLowerCase().slice(dotIndex);
}

function isAllowedFile(file, types, exts) {
  const ext = getFileExt(file.name);
  return types.includes(file.type) || exts.includes(ext);
}

export function isAllowedImage(file) {
  return isAllowedFile(file, IMAGE_TYPES, IMAGE_EXTS);
}

export function isAllowedVideo(file) {
  return isAllowedFile(file, VIDEO_TYPES, VIDEO_EXTS);
}

export function isAllowedVideoSize(file) {
  return file.size <= MAX_VIDEO_UPLOAD_MB * 1024 * 1024;
}
