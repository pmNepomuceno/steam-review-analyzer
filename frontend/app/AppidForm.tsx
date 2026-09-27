"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function AppidForm() {
  const router = useRouter();
  const [appid, setAppid] = useState("");
  return (
    <form
      className="appid"
      onSubmit={(e) => {
        e.preventDefault();
        // A number input also accepts "1e6" or "007"; normalize to plain digits.
        if (appid) router.push(`/games/${Number(appid)}`);
      }}
    >
      <label>
        Steam appid
        <input
          type="number"
          min={1}
          step={1}
          required
          placeholder="e.g. 1145360 (Hades)"
          value={appid}
          onChange={(e) => setAppid(e.target.value)}
        />
      </label>
      <button type="submit">Analyze</button>
    </form>
  );
}
