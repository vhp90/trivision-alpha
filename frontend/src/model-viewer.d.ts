import type { DetailedHTMLProps, HTMLAttributes } from "react";

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
        src?: string;
        poster?: string;
        alt?: string;
        ar?: boolean;
        autoplay?: boolean;
        cameraControls?: boolean;
        shadowIntensity?: string;
        exposure?: string;
        interactionPrompt?: string;
        environmentImage?: string;
      };
    }
  }
}
