import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const colors = {
  bg: "#f6fafc",
  ink: "#17212b",
  muted: "#5f7182",
  line: "#d7e3ec",
  blue: "#215a96",
  teal: "#0f766e",
  green: "#15845b",
  amber: "#b7791f",
  red: "#b42318",
  violet: "#6157d5",
  server: "#243447",
};

const sceneTitles = [
  {
    start: 0,
    end: 210,
    title: "Privacy-Preserving HAR Pipeline",
    subtitle:
      "30 wearable users: 29 train the global model, 1 is reserved as a new client.",
  },
  {
    start: 210,
    end: 450,
    title: "Local Training on Watches",
    subtitle:
      "Each watch keeps raw accelerometer, gyroscope, and activity labels on-device.",
  },
  {
    start: 450,
    end: 720,
    title: "Only Model Updates Travel",
    subtitle:
      "The server receives client_id, weights, and num_samples. Raw data is not uploaded.",
  },
  {
    start: 720,
    end: 960,
    title: "Server Aggregates with FedAvg",
    subtitle:
      "Client updates are averaged into a new global HAR model and sent back.",
  },
  {
    start: 960,
    end: 1200,
    title: "Project Output: Results + Privacy Evidence",
    subtitle:
      "Centralized and federated accuracy are compared while packet inspection proves the privacy claim.",
  },
  {
    start: 1200,
    end: 1500,
    title: "New Client Joins Later",
    subtitle:
      "The reserved user receives the global model, trains locally, sends one update, and joins the federation.",
  },
];

export const ProjectExplainer: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const title = sceneTitles.find((scene) => frame >= scene.start && frame < scene.end) ??
    sceneTitles[sceneTitles.length - 1];

  return (
    <AbsoluteFill style={styles.root}>
      <BackgroundGrid />
      <Header title={title.title} subtitle={title.subtitle} />
      <SystemCanvas frame={frame} fps={fps} />
      <Footer frame={frame} />
    </AbsoluteFill>
  );
};

const BackgroundGrid: React.FC = () => (
  <AbsoluteFill style={styles.background}>
    <div style={styles.blueWash} />
    <div style={styles.tealWash} />
  </AbsoluteFill>
);

const Header: React.FC<{title: string; subtitle: string}> = ({title, subtitle}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame % 240, [0, 18, 210, 238], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const y = interpolate(frame % 240, [0, 18], [-18, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{...styles.header, opacity, transform: `translateY(${y}px)`}}>
      <div style={styles.brand}>
        <div style={styles.brandMark}>FL</div>
        <div>
          <div style={styles.brandText}>Human Activity Recognition</div>
          <div style={styles.brandSub}>Federated learning project demo</div>
        </div>
      </div>
      <div style={styles.titleBlock}>
        <h1 style={styles.h1}>{title}</h1>
        <p style={styles.subtitle}>{subtitle}</p>
      </div>
    </div>
  );
};

const SystemCanvas: React.FC<{frame: number; fps: number}> = ({frame, fps}) => {
  const selected = frame >= 240 && frame < 720;
  const updatesActive = frame >= 450 && frame < 840;
  const aggregationActive = frame >= 720 && frame < 960;
  const outputActive = frame >= 960;
  const newClientActive = frame >= 1200;

  return (
    <div style={styles.canvas}>
      <div style={styles.leftColumn}>
        <PanelHeading label="Client watches" value="29 training clients" />
        <div style={styles.watchGrid}>
          {[1, 2, 3, 4, 5].map((id, index) => (
            <WatchClient
              key={id}
              id={id}
              active={selected && index < 4}
              locked={true}
              delay={index * 4}
            />
          ))}
        </div>
        <div style={styles.lockStrip}>
          <span style={styles.lockIcon}>LOCK</span>
          Raw sensor windows and labels stay on each watch
        </div>
      </div>

      <MovingUpdates active={updatesActive} />

      <div style={styles.centerColumn}>
        <PanelHeading label="Federated server" value="No raw data database" />
        <ServerRack active={aggregationActive} />
        <PayloadInspector active={updatesActive || aggregationActive} />
      </div>

      <MovingGlobalModel active={aggregationActive || outputActive} />

      <div style={styles.rightColumn}>
        <PanelHeading label="Outputs" value="Model + evidence" />
        <ResultCards active={outputActive} />
        <PrivacyComparison active={outputActive} />
      </div>

      <NewClientRibbon active={newClientActive} fps={fps} />
    </div>
  );
};

const PanelHeading: React.FC<{label: string; value: string}> = ({label, value}) => (
  <div style={styles.panelHeading}>
    <div style={styles.panelLabel}>{label}</div>
    <div style={styles.panelValue}>{value}</div>
  </div>
);

const WatchClient: React.FC<{
  id: number;
  active: boolean;
  locked: boolean;
  delay: number;
}> = ({id, active, locked, delay}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const bounce = spring({
    fps,
    frame: Math.max(0, frame - 215 - delay),
    config: {damping: 16, stiffness: 90},
  });
  const pulse = active
    ? interpolate(Math.sin((frame + delay) / 8), [-1, 1], [0.92, 1.04])
    : 1;

  return (
    <div
      style={{
        ...styles.watchCard,
        transform: `scale(${interpolate(bounce, [0, 1], [0.88, 1]) * pulse})`,
        borderColor: active ? "rgba(15,118,110,0.85)" : colors.line,
        background: active ? "#effbf8" : "#fbfdff",
      }}
    >
      <div style={styles.watchShape}>
        <div style={styles.watchBandTop} />
        <div style={styles.watchFace}>
          <div style={styles.watchGraph}>
            <span style={{height: 16}} />
            <span style={{height: 28}} />
            <span style={{height: 20}} />
            <span style={{height: 34}} />
          </div>
        </div>
        <div style={styles.watchBandBottom} />
      </div>
      <div>
        <div style={styles.clientName}>Watch {id}</div>
        <div style={styles.clientSub}>{active ? "Training locally" : "Private data"}</div>
        <div style={locked ? styles.goodTag : styles.warnTag}>Raw data local</div>
      </div>
    </div>
  );
};

const MovingUpdates: React.FC<{active: boolean}> = ({active}) => {
  const frame = useCurrentFrame();
  const opacity = active ? 1 : 0.2;

  return (
    <div style={styles.updateLane}>
      {[0, 1, 2].map((i) => {
        const x = active
          ? interpolate((frame * 3 + i * 95) % 285, [0, 285], [0, 1], {
              easing: Easing.inOut(Easing.ease),
            })
          : 0.15 + i * 0.2;
        return (
          <div
            key={i}
            style={{
              ...styles.packet,
              opacity,
              left: `${10 + x * 78}%`,
              top: 318 + i * 80,
            }}
          >
            weights
          </div>
        );
      })}
      <div style={styles.arrowLine} />
      <div style={styles.arrowLabel}>model updates only</div>
    </div>
  );
};

const ServerRack: React.FC<{active: boolean}> = ({active}) => {
  const frame = useCurrentFrame();
  const glow = active ? interpolate(Math.sin(frame / 6), [-1, 1], [0.18, 0.42]) : 0.1;

  return (
    <div style={{...styles.serverRack, boxShadow: `0 0 0 8px rgba(15,118,110,${glow})`}}>
      <div style={styles.serverTop}>FEDAVG SERVER</div>
      {[0, 1, 2].map((row) => (
        <div key={row} style={styles.serverSlot}>
          <div style={styles.serverLight} />
          <div style={styles.serverLine} />
          <div style={styles.serverLineSmall} />
        </div>
      ))}
      <div style={styles.formula}>Global = Σ weights x samples</div>
    </div>
  );
};

const PayloadInspector: React.FC<{active: boolean}> = ({active}) => (
  <div style={{...styles.payloadBox, opacity: active ? 1 : 0.62}}>
    <div style={styles.payloadTitle}>Server payload inspection</div>
    <code style={styles.codeLine}>client_id: 7</code>
    <code style={styles.codeLine}>weights: tensor state_dict</code>
    <code style={styles.codeLine}>num_samples: 246</code>
    <div style={styles.blockedLine}>raw_sensor_data: not present</div>
    <div style={styles.blockedLine}>activity_labels: not present</div>
  </div>
);

const MovingGlobalModel: React.FC<{active: boolean}> = ({active}) => {
  const frame = useCurrentFrame();
  const opacity = active ? 1 : 0.18;
  const x = active
    ? interpolate((frame * 2.4) % 240, [0, 240], [0, 1], {
        easing: Easing.inOut(Easing.ease),
      })
    : 0.2;

  return (
    <div style={styles.globalLane}>
      <div style={styles.arrowLine} />
      <div
        style={{
          ...styles.globalPacket,
          opacity,
          left: `${8 + x * 76}%`,
        }}
      >
        global model
      </div>
      <div style={styles.arrowLabel}>updated model distributed</div>
    </div>
  );
};

const ResultCards: React.FC<{active: boolean}> = ({active}) => {
  const frame = useCurrentFrame();
  const reveal = active
    ? spring({fps: 30, frame: frame - 960, config: {damping: 18, stiffness: 90}})
    : 0;

  return (
    <div style={{...styles.results, opacity: active ? 1 : 0.45}}>
      <Metric title="Centralized" value="95.5%" note="Raw data combined on server" accent={colors.blue} reveal={reveal} />
      <Metric title="Federated" value="91-95%" note="Model updates only" accent={colors.teal} reveal={reveal} />
      <Metric title="Privacy risk fields" value="0" note="No raw fields in FL packets" accent={colors.green} reveal={reveal} />
    </div>
  );
};

const Metric: React.FC<{
  title: string;
  value: string;
  note: string;
  accent: string;
  reveal: number;
}> = ({title, value, note, accent, reveal}) => (
  <div
    style={{
      ...styles.metric,
      transform: `translateY(${interpolate(reveal, [0, 1], [20, 0])}px)`,
      borderTopColor: accent,
    }}
  >
    <div style={styles.metricTitle}>{title}</div>
    <div style={{...styles.metricValue, color: accent}}>{value}</div>
    <div style={styles.metricNote}>{note}</div>
  </div>
);

const PrivacyComparison: React.FC<{active: boolean}> = ({active}) => (
  <div style={{...styles.privacyBox, opacity: active ? 1 : 0.35}}>
    <div style={styles.compareRow}>
      <div style={styles.badTag}>Centralized: raw_sensor_data + labels</div>
      <div style={styles.goodTag}>Federated: weights + sample count</div>
    </div>
    <div style={styles.privacyProof}>Communication-level privacy proof: raw client data is never uploaded in FL.</div>
  </div>
);

const NewClientRibbon: React.FC<{active: boolean; fps: number}> = ({active, fps}) => {
  const frame = useCurrentFrame();
  const slide = spring({
    fps,
    frame: active ? frame - 1200 : 0,
    config: {damping: 18, stiffness: 80},
  });

  return (
    <div
      style={{
        ...styles.newClientRibbon,
        transform: `translateY(${interpolate(slide, [0, 1], [180, 0])}px)`,
        opacity: active ? 1 : 0,
      }}
    >
      <div style={styles.newWatch}>
        <div style={styles.watchShapeSmall}>
          <div style={styles.watchFaceSmall}>30</div>
        </div>
      </div>
      <div>
        <div style={styles.newTitle}>Reserved Client 30 joins</div>
        <div style={styles.newSteps}>
          {"receives global model -> trains locally -> sends one model update -> server merges update"}
        </div>
      </div>
    </div>
  );
};

const Footer: React.FC<{frame: number}> = ({frame}) => {
  const progress = interpolate(frame, [0, 1499], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={styles.footer}>
      <div style={styles.footerText}>
        Dashboard explanation video | Centralized vs Federated Learning | 29 clients + 1 new client
      </div>
      <div style={styles.progressTrack}>
        <div style={{...styles.progressFill, width: `${progress}%`}} />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  root: {
    backgroundColor: colors.bg,
    color: colors.ink,
    fontFamily: "Inter, Segoe UI, Arial, sans-serif",
    overflow: "hidden",
  },
  background: {
    backgroundImage:
      "linear-gradient(rgba(33,90,150,0.055) 1px, transparent 1px), linear-gradient(90deg, rgba(33,90,150,0.055) 1px, transparent 1px)",
    backgroundSize: "54px 54px",
  },
  blueWash: {
    position: "absolute",
    width: 620,
    height: 620,
    borderRadius: 620,
    left: -180,
    top: -220,
    background: "rgba(33,90,150,0.13)",
    filter: "blur(30px)",
  },
  tealWash: {
    position: "absolute",
    width: 700,
    height: 700,
    borderRadius: 700,
    right: -210,
    top: -170,
    background: "rgba(15,118,110,0.12)",
    filter: "blur(36px)",
  },
  header: {
    position: "absolute",
    left: 76,
    right: 76,
    top: 48,
    display: "flex",
    justifyContent: "space-between",
    gap: 50,
    alignItems: "flex-start",
    zIndex: 3,
  },
  brand: {
    display: "flex",
    alignItems: "center",
    gap: 16,
  },
  brandMark: {
    width: 58,
    height: 58,
    borderRadius: 12,
    display: "grid",
    placeItems: "center",
    background: `linear-gradient(135deg, ${colors.blue}, ${colors.teal})`,
    color: "white",
    fontSize: 20,
    fontWeight: 900,
  },
  brandText: {
    fontSize: 22,
    fontWeight: 900,
  },
  brandSub: {
    fontSize: 14,
    color: colors.muted,
    marginTop: 4,
    fontWeight: 700,
  },
  titleBlock: {
    maxWidth: 980,
    textAlign: "right",
  },
  h1: {
    margin: 0,
    fontSize: 58,
    lineHeight: 1.02,
    letterSpacing: 0,
  },
  subtitle: {
    margin: "14px 0 0",
    fontSize: 25,
    lineHeight: 1.35,
    color: colors.muted,
    fontWeight: 600,
  },
  canvas: {
    position: "absolute",
    left: 76,
    right: 76,
    top: 190,
    bottom: 110,
    display: "grid",
    gridTemplateColumns: "450px 145px 450px 145px 450px",
    gap: 18,
    alignItems: "stretch",
  },
  leftColumn: {
    border: `1px solid ${colors.line}`,
    borderRadius: 18,
    background: "rgba(255,255,255,0.92)",
    boxShadow: "0 24px 58px rgba(30,45,60,0.11)",
    padding: 24,
  },
  centerColumn: {
    border: `1px solid ${colors.line}`,
    borderRadius: 18,
    background: "rgba(255,255,255,0.94)",
    boxShadow: "0 24px 58px rgba(30,45,60,0.11)",
    padding: 24,
  },
  rightColumn: {
    border: `1px solid ${colors.line}`,
    borderRadius: 18,
    background: "rgba(255,255,255,0.92)",
    boxShadow: "0 24px 58px rgba(30,45,60,0.11)",
    padding: 24,
  },
  panelHeading: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 22,
  },
  panelLabel: {
    fontSize: 28,
    fontWeight: 900,
  },
  panelValue: {
    color: colors.blue,
    background: "#e8f1f8",
    padding: "8px 12px",
    borderRadius: 999,
    fontSize: 15,
    fontWeight: 900,
  },
  watchGrid: {
    display: "grid",
    gap: 10,
  },
  watchCard: {
    display: "grid",
    gridTemplateColumns: "76px 1fr",
    alignItems: "center",
    gap: 14,
    border: `1px solid ${colors.line}`,
    borderRadius: 14,
    padding: 10,
  },
  watchShape: {
    position: "relative",
    height: 74,
    display: "grid",
    placeItems: "center",
  },
  watchBandTop: {
    position: "absolute",
    top: 0,
    width: 28,
    height: 17,
    borderRadius: "10px 10px 4px 4px",
    background: "#1f2d3a",
  },
  watchBandBottom: {
    position: "absolute",
    bottom: 0,
    width: 28,
    height: 17,
    borderRadius: "4px 4px 10px 10px",
    background: "#1f2d3a",
  },
  watchFace: {
    width: 56,
    height: 56,
    borderRadius: 15,
    border: "4px solid #1f2d3a",
    background: `linear-gradient(135deg, ${colors.blue}, ${colors.teal})`,
    display: "grid",
    placeItems: "center",
    zIndex: 1,
  },
  watchGraph: {
    height: 30,
    display: "flex",
    alignItems: "end",
    gap: 4,
  },
  clientName: {
    fontSize: 20,
    fontWeight: 900,
  },
  clientSub: {
    color: colors.muted,
    fontSize: 13,
    marginTop: 3,
    fontWeight: 700,
  },
  goodTag: {
    display: "inline-flex",
    marginTop: 7,
    padding: "5px 8px",
    borderRadius: 999,
    background: "#e2f6eb",
    color: "#075c3d",
    fontSize: 11,
    fontWeight: 900,
  },
  warnTag: {
    display: "inline-flex",
    marginTop: 10,
    padding: "6px 10px",
    borderRadius: 999,
    background: "#fff0d1",
    color: "#7a3d00",
    fontSize: 13,
    fontWeight: 900,
  },
  lockStrip: {
    marginTop: 14,
    borderRadius: 14,
    padding: 12,
    background: "#e8f7ef",
    color: "#075c3d",
    fontSize: 15,
    fontWeight: 900,
  },
  lockIcon: {
    display: "inline-flex",
    marginRight: 10,
    padding: "4px 8px",
    borderRadius: 6,
    background: "#075c3d",
    color: "white",
    fontSize: 12,
    verticalAlign: "middle",
  },
  updateLane: {
    position: "relative",
  },
  globalLane: {
    position: "relative",
  },
  arrowLine: {
    position: "absolute",
    left: 14,
    right: 14,
    top: "50%",
    height: 4,
    background: colors.line,
    borderRadius: 999,
  },
  arrowLabel: {
    position: "absolute",
    top: "calc(50% + 22px)",
    left: 0,
    right: 0,
    textAlign: "center",
    color: colors.blue,
    fontSize: 18,
    fontWeight: 900,
  },
  packet: {
    position: "absolute",
    transform: "translateX(-50%)",
    padding: "10px 14px",
    borderRadius: 999,
    background: colors.blue,
    color: "white",
    fontSize: 14,
    fontWeight: 900,
    boxShadow: "0 10px 24px rgba(33,90,150,0.22)",
  },
  globalPacket: {
    position: "absolute",
    top: "calc(50% - 24px)",
    transform: "translateX(-50%)",
    padding: "13px 16px",
    borderRadius: 999,
    background: colors.teal,
    color: "white",
    fontSize: 15,
    fontWeight: 900,
    boxShadow: "0 10px 24px rgba(15,118,110,0.22)",
  },
  serverRack: {
    borderRadius: 18,
    background: colors.server,
    color: "white",
    padding: 18,
    marginBottom: 18,
  },
  serverTop: {
    fontSize: 18,
    fontWeight: 900,
    marginBottom: 16,
  },
  serverSlot: {
    height: 70,
    borderRadius: 12,
    background: "#314558",
    marginBottom: 12,
    display: "flex",
    alignItems: "center",
    gap: 14,
    padding: "0 14px",
  },
  serverLight: {
    width: 14,
    height: 14,
    borderRadius: 14,
    background: "#32d583",
    boxShadow: "0 0 18px rgba(50,213,131,0.8)",
  },
  serverLine: {
    width: 165,
    height: 8,
    borderRadius: 999,
    background: "#8da2b3",
  },
  serverLineSmall: {
    width: 78,
    height: 8,
    borderRadius: 999,
    background: "#647789",
  },
  formula: {
    background: "#101820",
    color: "#d8f8ff",
    borderRadius: 12,
    padding: 13,
    fontSize: 18,
    fontFamily: "Consolas, Courier New, monospace",
  },
  payloadBox: {
    border: `1px solid ${colors.line}`,
    borderRadius: 16,
    background: "#fbfdff",
    padding: 16,
  },
  payloadTitle: {
    fontWeight: 900,
    fontSize: 20,
    marginBottom: 12,
  },
  codeLine: {
    display: "block",
    background: "#102032",
    color: "#d8f8ff",
    padding: "9px 10px",
    borderRadius: 8,
    marginBottom: 8,
    fontSize: 15,
  },
  blockedLine: {
    color: "#075c3d",
    background: "#e2f6eb",
    borderRadius: 8,
    padding: "9px 10px",
    marginBottom: 8,
    fontSize: 15,
    fontWeight: 900,
  },
  results: {
    display: "grid",
    gap: 16,
  },
  metric: {
    border: `1px solid ${colors.line}`,
    borderTop: `7px solid ${colors.blue}`,
    borderRadius: 16,
    background: "#fbfdff",
    padding: 18,
  },
  metricTitle: {
    color: colors.muted,
    fontSize: 17,
    fontWeight: 900,
  },
  metricValue: {
    fontSize: 46,
    fontWeight: 900,
    marginTop: 8,
  },
  metricNote: {
    color: colors.muted,
    fontSize: 16,
    fontWeight: 700,
    marginTop: 6,
  },
  privacyBox: {
    marginTop: 18,
    border: `1px solid ${colors.line}`,
    borderRadius: 16,
    padding: 16,
    background: "#ffffff",
  },
  compareRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 10,
  },
  badTag: {
    display: "inline-flex",
    padding: "8px 11px",
    borderRadius: 999,
    background: "#ffe4df",
    color: "#8d1b12",
    fontSize: 14,
    fontWeight: 900,
  },
  privacyProof: {
    marginTop: 15,
    color: colors.ink,
    fontSize: 18,
    fontWeight: 900,
    lineHeight: 1.35,
  },
  newClientRibbon: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    minHeight: 138,
    border: `1px solid ${colors.line}`,
    borderRadius: 18,
    background: "#ffffff",
    boxShadow: "0 24px 58px rgba(30,45,60,0.13)",
    display: "flex",
    alignItems: "center",
    gap: 24,
    padding: 22,
  },
  newWatch: {
    width: 100,
  },
  watchShapeSmall: {
    width: 78,
    height: 92,
    borderRadius: 28,
    background: "#1f2d3a",
    display: "grid",
    placeItems: "center",
  },
  watchFaceSmall: {
    width: 62,
    height: 62,
    borderRadius: 18,
    display: "grid",
    placeItems: "center",
    background: `linear-gradient(135deg, ${colors.violet}, ${colors.teal})`,
    color: "white",
    fontSize: 23,
    fontWeight: 900,
  },
  newTitle: {
    fontSize: 30,
    fontWeight: 900,
  },
  newSteps: {
    marginTop: 8,
    fontSize: 20,
    color: colors.muted,
    fontWeight: 800,
  },
  footer: {
    position: "absolute",
    left: 76,
    right: 76,
    bottom: 42,
  },
  footerText: {
    fontSize: 16,
    color: colors.muted,
    fontWeight: 800,
    marginBottom: 10,
  },
  progressTrack: {
    height: 8,
    borderRadius: 999,
    background: "#d7e3ec",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    borderRadius: 999,
    background: `linear-gradient(90deg, ${colors.blue}, ${colors.teal})`,
  },
};
