import AppidForm from "./AppidForm";

export default function Home() {
  return (
    <>
      <h1>Steam Review Analyzer</h1>
      <p className="muted">
        Enter a game&apos;s appid (the number in its store URL,{" "}
        <code>store.steampowered.com/app/&lt;appid&gt;</code>). A game seen for the first time is
        fetched from Steam and analyzed, which takes up to a minute.
      </p>
      <AppidForm />
    </>
  );
}
