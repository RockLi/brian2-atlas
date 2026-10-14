# Preserved upstream workflows

These are the exact workflow files inherited from official Brian2 at the migration baseline. They are stored outside `.github/workflows/`, so Atlas pushes do not trigger upstream publication, Docker publication or upstream-specific scheduled maintenance. Atlas validation runs from `../workflows/atlas.yml`. Public release workflows will be configured separately from the migration goal.
