import { useEffect, useState } from "react";
import FileUpload from "../components/FileUpload";
import WindowSelector from "../components/WindowSelector";
import RiskTable from "../components/RiskTable";
import CareROI from "../components/CareROI";
import { getPredictions, getCareSimulation } from "../api/backend";

export default function Dashboard() {
  const [window, setWindow] = useState(30); // ✅ number
  const [predictions, setPredictions] = useState([]);
  const [careData, setCareData] = useState([]);
  const [dataLoaded, setDataLoaded] = useState(false);

  // -------------------- Load AFTER upload --------------------
  const loadResults = async () => {
    const pred = await getPredictions(window);
    const care = await getCareSimulation(window);

    setPredictions(pred.sample_predictions);
    setCareData(care.care_simulation_results);
    setDataLoaded(true);
  };

  // -------------------- Refresh WHEN window changes --------------------
  useEffect(() => {
    if (!dataLoaded) return;

    const refreshByWindow = async () => {
      const pred = await getPredictions(window);
      const care = await getCareSimulation(window);

      setPredictions(pred.sample_predictions);
      setCareData(care.care_simulation_results);
    };

    refreshByWindow();
  }, [window, dataLoaded]);

  return (
    <div>
      <h1>Member Risk Stratification Dashboard</h1>

      <FileUpload onComplete={loadResults} />

      <WindowSelector window={window} setWindow={setWindow} />

      <h2>Risk Stratification ({window}-day)</h2>

      <RiskTable key={`risk-${window}`} data={predictions} />

      <CareROI key={`care-${window}`} data={careData} />
    </div>
  );
}
