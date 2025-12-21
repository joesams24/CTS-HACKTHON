export default function WindowSelector({ window, setWindow }) {
  return (
    <div>
      <label>Prediction Window: </label>
      <select
        value={window}
        onChange={(e) => setWindow(Number(e.target.value))} // ✅ FIX
      >
        <option value={30}>30 Days</option>
        <option value={60}>60 Days</option>
        <option value={90}>90 Days</option>
      </select>
    </div>
  );
}
