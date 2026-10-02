import GameBrowser from "./GameBrowser";

export default function Home() {
  return (
    <>
      <div className="intro">
        <h1>What Steam reviews say, topic by topic</h1>
        <p>
          Steam gives every game one thumbs-up percentage. This splits a game&apos;s most recent
          reviews into sentences, sorts them into performance, price, bugs, story and gameplay,
          and shows how players feel about each one, with their own words as examples.
        </p>
      </div>
      <GameBrowser />
    </>
  );
}
