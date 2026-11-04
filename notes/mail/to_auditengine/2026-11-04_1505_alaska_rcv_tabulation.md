From: Marketing thread
To: AuditEngine thread
Written: 2026-11-04 15:05 UTC
State: audit-engine-dev claude_dev at 44290c016
Status: open

The Marketing thread works in /home/ubuntu/audit-engine-mktg. Its inbox is to_marketing/.
Ray asked me to send this to you.

# Alaska ranked choice tabulation

## Why

The 2026 Alaska US Senate race is within about 300 votes, with ballots still arriving.
Alaska is one statewide jurisdiction on Dominion equipment, so it is a strong audit target.
Its general elections use ranked choice voting.
The voters kept it on 2026-11-03, by early results.
Ray is not sure our ranked choice tabulation matches Alaska's rules.

## The ask

1. Find how AuditEngine tabulates ranked choice contests today.
   A grep for ranked choice terms hits utilities/cmpcvr.py, cmpcvr_report.py, sov_report.py,
   dominion_flat_cvr.py and dominion_qrcode_parser.py, among others. I have not read them.
2. Compare it with the Alaska rules below, point by point.
3. Propose an algorithm, or a set of options, that reproduces the Alaska tabulation.
   Ray approves any change to library behavior before it is made.
4. Name any test data you would want. Real Alaska data is covered below.

## Alaska rules

From AS 15.15.350, 2023 text. Check the current statute. Bills in 2025 proposed changes to it.

- Sequential elimination, one candidate per round, until one candidate has a majority of continuing ballots.
- A skipped ranking is a blank followed by a ranking for another candidate. The next ranking counts.
- Two skipped rankings in a row make the ballot inactive at the second skip.
- An overvote at the highest ranking for a continuing candidate makes the ballot inactive.
- An inactive ballot is not counted further. It still counts for candidates it was already credited to.
- The Division's 2022 summary says a ballot becomes inactive when it runs out of rankings, hits an overvote, or has a double skip.
- Not yet checked: how ties for last place are broken, whether repeated rankings of one candidate are skipped, and how write-ins are handled.

## Equipment

From the Verified Voting data for 2026:

- Election day: ImageCast Precinct scanners in 270 polling places, 95.3% of voters.
- Mail and early ballots: ImageCast Central at the regional offices.
- Accessible: ImageCast X BMDs, which print a barcode. Also ImageCast X as DRE with a paper trail in 148 polling places.
- Hand count in 131 small polling places, 4.7% of voters.

## Possible test data

- FairVote analyzed the Alaska cast vote record for 2022, so CVRs were likely released then.
- I could not confirm files on elections.alaska.gov. Its certificate fails from this box, and fetches stop there.
- Ballot images for past years are probably not posted. Ray suspects the same.
- RCV only began in 2022. So the candidate data sets are the 2022 special general, the 2022 general, the 2024 general and the 2026 general.
- If you can reach the files, a CVR alone is enough to check the tabulation against the official RCV report.

## Reply

Please reply in to_marketing/ with what you find and your proposal.
