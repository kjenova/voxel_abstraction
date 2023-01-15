function preprocessing_analysis()
    erosion_analysis();
    resizing_analysis();
    normals_analysis();
end

function [] = erosion_analysis()
    startup;
    globals;
    cachedir = cachedir;

    erosionDir = fullfile(cachedir, 'analysis', 'erosion');

    maxKernelSize = 6;
    nTopComponents = 10;

    V = niftiread('mito-endolyso.nii');
    V = V == 1;

    for kernelSize = 1:maxKernelSize
        if kernelSize > 1
            E = imerode(V, strel("sphere", kernelSize));
        else
            E = V;
        end

        CC = bwconncomp(E);
        numPixels = cellfun(@numel, CC.PixelIdxList);
        [~, indices] = sort(numPixels, 'descend');

        for i = 1:nTopComponents
            X = zeros(size(V));
            X(CC.PixelIdxList{indices(i)}) = 1;
            cropped = regionprops(X, "FilledImage").FilledImage;
            [FV, ~] = Voxel2mesh(cropped);
            faces = FV.faces;

            maxSize = max(size(cropped));
            normalizedVertices = (FV.vertices - 0.5) / maxSize;
            vertices = normalizedVertices - 0.5;

            erosionFile = fullfile(erosionDir, strcat(num2str(i), '_', num2str(kernelSize), '.mat'));
            save(erosionFile, 'vertices', 'faces');
        end
    end
end

function [] = resizing_analysis()
    startup;
    globals;
    cachedir = cachedir;

    resizingDir = fullfile(cachedir, 'analysis', 'resizing');

    % Tole je zadnja oblika v testni množici. Je najbolj "kompleksna".
    % V = niftiread('branched/fib1-4-3-0.nii.gz');
    % Ampak še vedno izgleda dobro pri gridSize = 32. Druga komponenta v učni množici pa ne:
    V = niftiread('mito-endolyso.nii');
    V = V == 1;
    V = imerode(V, strel("sphere", 3));
    CC = bwconncomp(V);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, indices] = sort(numPixels, 'descend');

    X = zeros(size(V));
    % X(CC.PixelIdxList{indices(1)}) = 1;
    X(CC.PixelIdxList{indices(2)}) = 1;
    cropped = regionprops(X, "FilledImage").FilledImage;

    [FV, ~] = Voxel2mesh(cropped);
    faces = FV.faces;

    maxSize = max(size(cropped));
    normalizedVertices = (FV.vertices - 0.5) / maxSize;
    vertices = normalizedVertices - 0.5;

    resizingFile = fullfile(resizingDir, 'original.mat');
    save(resizingFile, 'vertices', 'faces')

    resizing_analysis_helper(cropped, 32, resizingDir);
    resizing_analysis_helper(cropped, 64, resizingDir);
end

function Volume = resize_volume(Volume, gridSize)
    maxSize = max(size(Volume));
    Volume = imresize3(Volume, gridSize / maxSize);
    padding = gridSize - size(Volume);
    padding_pre = floor(padding / 2);
    Volume = padarray(Volume, padding_pre, 0, 'pre');
    Volume = padarray(Volume, padding - padding_pre, 0, 'post');
    if ~isequal(size(Volume), [gridSize gridSize gridSize])
        disp("!!!!!!!!");
    end
end

function [faces, vertices] = resizing_analysis_helper(cropped, gridSize, resizingDir)
    Volume = resize_volume(cropped, gridSize);

    [FV, ~] = Voxel2mesh(Volume);
    faces = FV.faces;

    normalizedVertices = (FV.vertices - 0.5) / gridSize;
    vertices = normalizedVertices - 0.5;

    resizingFile = fullfile(resizingDir, strcat(num2str(gridSize), '.mat'));
    save(resizingFile, 'vertices', 'faces');
end

function [] = normals_analysis()
    startup;
    globals;
    cachedir = cachedir;

    numSamples = 100;

    normalsDir = fullfile(cachedir, 'analysis', 'normals');

    % Tole je šesta oblika v testni množici. Je "preprosta".
    V = niftiread('branched/fib1-3-2-1.nii.gz');
    V = V == 2;
    CC = bwconncomp(V);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, indices] = sort(numPixels, 'descend');

    X = zeros(size(V));
    X(CC.PixelIdxList{indices(19)}) = 1;
    cropped = regionprops(X, "FilledImage").FilledImage;
    [FV, ~] = Voxel2mesh(cropped);
    faces = FV.faces;

    maxSize = max(size(cropped));
    normalizedVertices = (FV.vertices - 0.5) / maxSize;
    vertices = normalizedVertices - 0.5;

    surfaceSamples = sampling(FV.faces, vertices, numSamples);
    points = surfaceSamples(:, 1:3);
    normals = surfaceSamples(:, 4:6);

    normalsFile = fullfile(normalsDir, strcat(num2str(1), '.mat'));
    save(normalsFile, 'vertices', 'faces', 'points', 'normals');
end

function [finalSamples] = sampling(faces, vertices, numSamples)
    samples = basic_sampling(faces, vertices, numSamples);

    finalSamples = zeros(numSamples, 6);
    finalSamples(:, 1:3) = samples(:, 1:3);

    faceCenters = face_centers(faces, vertices);

    k = 6;
    D = pdist2(samples(:, 1:3), faceCenters(:, 1:3));
    for sId = 1:numSamples
        normal = zeros(1, 3);
        [B, I] = mink(D(sId, :), k);
        B = 1 ./ B;
        B = B ./ sum(B);
        for j = 1:k
            normal = normal + B(j) * faceCenters(I(j), 4:6);
        end
        normal = normal / norm(normal);
        finalSamples(sId, 4:6) = normal;
    end
end

function [samples] = basic_sampling(faces, vertices, numSamples)
    samples = zeros(numSamples, 6);
    faceIds = datasample(1:size(faces, 1), numSamples);
    paras = rand(2, numSamples);

    for sId = 1:numSamples
        faceId = faceIds(sId);
        p1 = vertices(faces(faceId, 1), :);
        p2 = vertices(faces(faceId, 2), :);
        p3 = vertices(faces(faceId, 3), :);
        
        r1 = paras(1, sId);
        r2 = paras(2, sId);
        t1 = 1-sqrt(r1);
        t2 = sqrt(r1)*(1-r2);
        t3 = sqrt(r1)*r2;
        samples(sId, 1:3) = t1*p1 + t2*p2 + t3*p3;

        c = cross(p2 - p1, p3 - p1);
        samples(sId, 4:6) = c / norm(c);
    end
end

function [centers] = face_centers(faces, vertices)
    centers = zeros(size(faces, 1), 6);
    for faceId = 1:size(faces, 1)
        p1 = vertices(faces(faceId, 1), :);
        p2 = vertices(faces(faceId, 2), :);
        p3 = vertices(faces(faceId, 3), :);

        centers(faceId, 1:3) = (p1 + p2 + p3) / 3;
        c = cross(p2 - p1, p3 - p1);
        centers(faceId, 4:6) = c / norm(c);
    end
end
