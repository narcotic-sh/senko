import "./styles.css";

const root = document.querySelector<HTMLElement>("#app");
if (root === null) {
  throw new Error("Missing #app root element");
}

const search = new URLSearchParams(location.search);
if (search.has("dual-device-concurrency-diagnostic")) {
  void import("./dual-device-concurrency-diagnostic").then(
    ({ runDualDeviceConcurrencyDiagnostic }) =>
      runDualDeviceConcurrencyDiagnostic(root),
  );
} else if (search.has("raw-campplus-graph-diagnostic")) {
  void import("./raw-campplus-graph-diagnostic").then(
    ({ runRawCampPlusGraphDiagnostic }) => runRawCampPlusGraphDiagnostic(root),
  );
} else if (search.has("raw-campplus-file-parity")) {
  void import("./raw-campplus-file-parity").then(
    ({ runRawCampPlusFileParityDiagnostic }) =>
      runRawCampPlusFileParityDiagnostic(root),
  );
} else if (search.has("raw-campplus-dense-diagnostic")) {
  void import("./raw-campplus-dense-diagnostic").then(
    ({ runRawCampPlusDenseDiagnostic }) => runRawCampPlusDenseDiagnostic(root),
  );
} else if (search.has("raw-campplus-diagnostic")) {
  void import("./raw-campplus-diagnostic").then(({ runRawCampPlusDiagnostic }) =>
    runRawCampPlusDiagnostic(root),
  );
} else {
  void import("./app").then(({ SenkoBrowserApp }) => {
    const app = new SenkoBrowserApp(root);
    return app.start();
  });
}
