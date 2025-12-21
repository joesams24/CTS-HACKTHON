export default function RiskTable({ data }) {
  return (
    <table border="1">
      <thead>
        <tr>
          <th>Risk Probability</th>
          <th>Risk Score</th>
          <th>Risk Tier</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, idx) => (
          <tr key={idx}>
            <td>{row.risk_probability}</td>
            <td>{row.risk_score}</td>
            <td>{row.risk_tier}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
