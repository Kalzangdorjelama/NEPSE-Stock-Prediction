export function LoadingSpinner() {
  return (
    <div
      style={{
        border: "4px solid #f3f3f3",
        borderTop: "4px solid #3498db",
        borderRadius: "50%",
        width: "40px",
        height: "40px",
        animation: "spin 1s linear infinite",
        margin: "auto",
      }}
    />
  );
}

const styles = `
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
`;

const StyleInjector = () => <style>{styles}</style>;

export default function App() {
  return (
    <div className="App">
      <StyleInjector />
      <h2>Loading...</h2>
      <LoadingSpinner />
    </div>
  );
}
