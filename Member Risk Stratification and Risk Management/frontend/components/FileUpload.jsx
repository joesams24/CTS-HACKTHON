import { useState } from "react";
import { uploadFile, trainModel } from "../api/backend";

export default function FileUpload({ onComplete }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) return;

    setStatus("Uploading...");
    await uploadFile(file);

    setStatus("Training model...");
    await trainModel();

    setStatus("Ready");
    onComplete();
  };

  return (
    <div>
      <h3>Upload Patient Dataset</h3>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload}>Upload & Train</button>
      <p>{status}</p>
    </div>
  );
}
