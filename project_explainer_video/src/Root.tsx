import "./index.css";
import { Composition } from "remotion";
import { ProjectExplainer } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="PrivacyPipelineExplainer"
        component={ProjectExplainer}
        durationInFrames={1500}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
