# Grad ML

Source code for [https://gradml.mit.edu](https://gradml.mit.edu)

Documentations for:

- [Content Creation](/_docs/_contents.md)
- [Developers](/_docs/_devs.md)
- [Staff members](/_docs/_staff.md)


# Jagdeep's Guide for Adding HWs & Lectures

### Step 1: Add Asset Files
Add hw pdfs or lecture slide/notes to `assets\homeworks` or `assets\lectures` respectively.

### Step 2: Edit Card
Make a copy of the card
`_homeworks\hw0.md`
or 
`_lectures\lecture1.md`
for your new hw/lecture.

**For Homeworks**
- `title`, `release_date`, `due_date` are required fields.
- The other fields will show `to be released` if not provided.
- Use relative paths to link assets.
- Note that the sorting of homeworks is based on `release_date`

**For Lectures**
- `title` and `id` are required fields
- lecture_`n` should have `id: n` (used for sorting)
- The other fields will show `to be released` if not provided.
- Use relative paths to link assets.

### Step 3: Publish
By default, only the first `n` hws and `m` lectures are displayed on the homeworks and lectures pages. To increase this number (and publish a new hw/lecture) navigate to 
`main\homeworks.md` or `main\lectures.md` and increase the `limit_value` variable.


